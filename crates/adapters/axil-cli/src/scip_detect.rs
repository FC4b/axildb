//! Polyglot SCIP project detection for `axil scip refresh` / `status` /
//! `doctor`.
//!
//! SCIP indexers must run from the directory that holds the project's own
//! config (`tsconfig.json`, `pyproject.toml`, …), not from the repo root.
//! A monorepo with `frontend/package.json` and `backend/pyproject.toml`
//! has no root marker at all, so a root-only peek detects nothing. This
//! module walks the tree (depth- and count-bounded so the brain hook's
//! `--if-stale` fast path stays cheap) and returns one [`ScipProject`]
//! per `(language, project dir)` pair.
//!
//! The output-naming, age, and labeling helpers live here too so
//! `refresh` and `status` cannot drift apart on where a project's index
//! is expected to be.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

/// Marker files → SCIP-indexable language, in suggestion-priority order.
/// The order doubles as the tiebreak for which language `axil doctor`
/// lists first, so keep rust → python → typescript → go → java.
const MARKERS: &[(&str, &str)] = &[
    ("Cargo.toml", "rust"),
    ("pyproject.toml", "python"),
    ("setup.py", "python"),
    ("package.json", "typescript"),
    ("go.mod", "go"),
    ("pom.xml", "java"),
    ("build.gradle", "java"),
];

/// Directory names never descended into. Dot-prefixed dirs are skipped
/// unconditionally (covers `.git`, `.venv`, `.axil`, `.next`, …).
const SKIP_DIRS: &[&str] = &[
    "node_modules",
    "target",
    "dist",
    "build",
    "vendor",
    "venv",
    "env",
    "__pycache__",
    "coverage",
];

/// How deep below the repo root to look for marker files. Depth 1 is a
/// direct child (`frontend/package.json`); 4 covers `apps/web/ui/pkg`.
const MAX_DEPTH: usize = 4;

/// Hard cap on directories visited, so a pathological tree can't make
/// the brain hook's first-tool-call check slow.
const MAX_DIRS: usize = 2_000;

/// One indexable project: a language plus the directory its SCIP indexer
/// must run from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScipProject {
    pub language: &'static str,
    /// Always absolute — [`detect_scip_projects`] absolutizes its root.
    pub dir: PathBuf,
}

/// Make `p` absolute against the current dir. No canonicalization — all
/// callers build and compare paths through this same function, so
/// symlink divergence can't split them. A relative `--db` path would
/// otherwise derive an empty repo root whose `read_dir` silently fails.
pub fn absolutize(p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|cwd| cwd.join(p))
            .unwrap_or_else(|_| p.to_path_buf())
    }
}

/// Walk `repo_root` and return every `(language, project dir)` pair.
///
/// Breadth-first, so shallower projects come first; within one directory,
/// [`MARKERS`] order applies. A project nested inside an already-found
/// project of the *same* language is skipped (Cargo/npm workspace members
/// roll up to the workspace root), while a different language nested in
/// another's tree is kept (e.g. a Python tool inside a Rust repo).
pub fn detect_scip_projects(repo_root: &Path) -> Vec<ScipProject> {
    let repo_root = absolutize(repo_root);
    let mut projects: Vec<ScipProject> = Vec::new();
    let mut queue: VecDeque<(PathBuf, usize)> = VecDeque::new();
    queue.push_back((repo_root.clone(), 0));
    let mut visited = 0usize;

    while let Some((dir, depth)) = queue.pop_front() {
        visited += 1;
        if visited > MAX_DIRS {
            break;
        }

        for (file, lang) in MARKERS {
            if !dir.join(file).is_file() {
                continue;
            }
            // BFS visits each dir once and starts_with covers equality,
            // so this also dedups a second marker of the same language
            // in one dir (pyproject.toml + setup.py).
            let nested_in_same_lang = projects
                .iter()
                .any(|p| p.language == *lang && dir.starts_with(&p.dir));
            if !nested_in_same_lang {
                projects.push(ScipProject {
                    language: lang,
                    dir: dir.clone(),
                });
            }
        }

        if depth >= MAX_DEPTH {
            continue;
        }
        let entries = match std::fs::read_dir(&dir) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.starts_with('.') || SKIP_DIRS.contains(&name.as_ref()) {
                continue;
            }
            queue.push_back((entry.path(), depth + 1));
        }
    }
    filter_gitignored(&repo_root, projects)
}

/// Drop subfolder projects whose dir is gitignored — experiment
/// sandboxes, vendored clones, generated trees. One `git check-ignore
/// --stdin` call for all candidates; best-effort, so a missing `git`
/// binary or a non-git directory filters nothing. The repo-root
/// project itself is never filtered.
fn filter_gitignored(repo_root: &Path, projects: Vec<ScipProject>) -> Vec<ScipProject> {
    use std::io::Write;

    if !projects.iter().any(|p| p.dir != repo_root) {
        return projects;
    }
    let Ok(mut child) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo_root)
        .args(["check-ignore", "--stdin", "-z"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        return projects;
    };
    if let Some(stdin) = child.stdin.take() {
        let mut stdin = std::io::BufWriter::new(stdin);
        for p in &projects {
            if p.dir != repo_root {
                let _ = stdin.write_all(p.dir.to_string_lossy().as_bytes());
                let _ = stdin.write_all(b"\0");
            }
        }
    }
    let Ok(out) = child.wait_with_output() else {
        return projects;
    };
    // Exit 128 = not a git repo / bad usage; 0/1 = ran fine with/without
    // matches. Only trust the output in the ran-fine cases.
    if !matches!(out.status.code(), Some(0) | Some(1)) {
        return projects;
    }
    let ignored: std::collections::HashSet<PathBuf> = out
        .stdout
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| PathBuf::from(String::from_utf8_lossy(s).into_owned()))
        .collect();
    projects
        .into_iter()
        .filter(|p| !ignored.contains(&p.dir))
        .collect()
}

/// Unique languages across the detected projects, preserving discovery
/// order (root markers first, then by depth).
pub fn detected_languages(projects: &[ScipProject]) -> Vec<&'static str> {
    let mut langs: Vec<&'static str> = Vec::new();
    for p in projects {
        if !langs.contains(&p.language) {
            langs.push(p.language);
        }
    }
    langs
}

/// Map a user-supplied `--language` value to its canonical `&'static
/// str`, or None when unsupported. One lookup serves validation and
/// static-str narrowing, so the supported set lives only in [`MARKERS`].
pub fn normalize_language(lang: &str) -> Option<&'static str> {
    MARKERS
        .iter()
        .map(|(_, l)| *l)
        .find(|l| *l == lang)
}

/// Seconds since `path`'s mtime; None when the file is missing,
/// unreadable, or future-dated (clock skew).
pub fn age_secs(path: &Path) -> Option<u64> {
    std::fs::metadata(path)
        .ok()?
        .modified()
        .ok()?
        .elapsed()
        .ok()
        .map(|e| e.as_secs())
}

/// Human label for a project dir: path relative to the repo root, or
/// "." for the root project.
pub fn rel_label(dir: &Path, repo_root: &Path) -> String {
    absolutize(dir)
        .strip_prefix(absolutize(repo_root))
        .ok()
        .filter(|p| !p.as_os_str().is_empty())
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| ".".to_string())
}

/// Where `scip refresh` writes a project's index, and where `status`
/// and freshness checks must look for it. Single-project repos keep the
/// legacy `index.scip` name (zero migration); polyglot repos get one
/// file per project via [`output_file_name`]. Anchored at the `.axil`
/// dir so custom `--db` layouts resolve consistently everywhere.
pub fn expected_output(
    project: &ScipProject,
    polyglot: bool,
    axil_dir: &Path,
    repo_root: &Path,
) -> PathBuf {
    let axil_dir = absolutize(axil_dir);
    if polyglot {
        axil_dir.join(output_file_name(project, repo_root))
    } else {
        axil_dir.join("index.scip")
    }
}

/// Stable 64-bit FNV-1a — `DefaultHasher` is not guaranteed stable
/// across Rust releases, and a name change would silently re-index
/// every subfolder project after a toolchain bump.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// File name for a project's generated SCIP index inside `.axil/`.
///
/// Root project → `index-<lang>.scip`; subfolder project →
/// `index-<lang>-<dir slug>-<hash>.scip`. The slug is lossy (case
/// folding, non-alphanumerics collapse to `-`), so the short hash of
/// the exact relative path is what guarantees sibling projects like
/// `web-ui` / `web.ui` / `Web UI` never share an output file. The name
/// depends only on `(language, dir)`, so it is stable as other
/// projects come and go.
pub fn output_file_name(project: &ScipProject, repo_root: &Path) -> String {
    let dir = absolutize(&project.dir);
    let root = absolutize(repo_root);
    let rel = dir.strip_prefix(&root).unwrap_or(&dir);
    if rel.as_os_str().is_empty() {
        return format!("index-{}.scip", project.language);
    }
    let rel_str = rel.to_string_lossy();
    let mut slug = String::new();
    for c in rel_str.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
        } else if !slug.ends_with('-') {
            slug.push('-');
        }
    }
    let slug = slug.trim_matches('-');
    let hash = fnv1a(rel_str.as_bytes()) & 0xff_ffff;
    if slug.is_empty() {
        format!("index-{}-{hash:06x}.scip", project.language)
    } else {
        format!("index-{}-{slug}-{hash:06x}.scip", project.language)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn touch(path: &Path) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, b"").unwrap();
    }

    #[test]
    fn detects_subfolder_only_polyglot_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        touch(&root.join("frontend/package.json"));
        touch(&root.join("backend/pyproject.toml"));

        let projects = detect_scip_projects(root);
        let mut pairs: Vec<(&str, PathBuf)> = projects
            .iter()
            .map(|p| (p.language, p.dir.clone()))
            .collect();
        pairs.sort();
        assert_eq!(
            pairs,
            vec![
                ("python", root.join("backend")),
                ("typescript", root.join("frontend")),
            ]
        );
    }

    #[test]
    fn root_marker_detected_with_nested_same_language_deduped() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        touch(&root.join("Cargo.toml"));
        // Workspace member must roll up to the root project.
        touch(&root.join("crates/axil-core/Cargo.toml"));
        // Different language nested in the tree is kept.
        touch(&root.join("crates/axil-ui/package.json"));

        let projects = detect_scip_projects(root);
        assert_eq!(projects.len(), 2);
        assert_eq!(projects[0].language, "rust");
        assert_eq!(projects[0].dir, root);
        assert_eq!(projects[1].language, "typescript");
        assert_eq!(projects[1].dir, root.join("crates/axil-ui"));
    }

    #[test]
    fn sibling_projects_of_same_language_both_detected() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        touch(&root.join("frontend/package.json"));
        touch(&root.join("admin/package.json"));

        let projects = detect_scip_projects(root);
        assert_eq!(projects.len(), 2);
        assert!(projects.iter().all(|p| p.language == "typescript"));
    }

    #[test]
    fn two_markers_of_same_language_in_one_dir_dedup() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        touch(&root.join("api/pyproject.toml"));
        touch(&root.join("api/setup.py"));

        let projects = detect_scip_projects(root);
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].language, "python");
    }

    #[test]
    fn skip_dirs_and_dot_dirs_are_not_descended() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        touch(&root.join("node_modules/leftpad/package.json"));
        touch(&root.join("target/debug/Cargo.toml"));
        touch(&root.join(".hidden/go.mod"));

        assert!(detect_scip_projects(root).is_empty());
    }

    #[test]
    fn markers_beyond_max_depth_are_ignored() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        touch(&root.join("a/b/c/d/e/package.json"));
        assert!(detect_scip_projects(root).is_empty());

        touch(&root.join("a/b/c/d/package.json"));
        assert_eq!(detect_scip_projects(root).len(), 1);
    }

    #[test]
    fn one_dir_can_host_two_languages() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        touch(&root.join("Cargo.toml"));
        touch(&root.join("package.json"));

        let projects = detect_scip_projects(root);
        assert_eq!(detected_languages(&projects), vec!["rust", "typescript"]);
        assert!(projects.iter().all(|p| p.dir == root));
    }

    #[test]
    fn relative_repo_root_is_absolutized() {
        // A relative --db used to derive an empty repo root whose
        // read_dir silently failed, hiding subfolder projects.
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        touch(&root.join("frontend/package.json"));

        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(root).unwrap();
        let projects = detect_scip_projects(Path::new(""));
        std::env::set_current_dir(prev).unwrap();

        assert_eq!(projects.len(), 1);
        assert!(projects[0].dir.is_absolute());
    }

    #[test]
    fn gitignored_subprojects_are_filtered() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        touch(&root.join("app/package.json"));
        touch(&root.join("sandbox/package.json"));
        std::fs::write(root.join(".gitignore"), "sandbox/\n").unwrap();
        let git_ready = std::process::Command::new("git")
            .arg("-C")
            .arg(root)
            .args(["init", "-q"])
            .status()
            .map(|s| s.success())
            .unwrap_or(false);
        if !git_ready {
            // No git on PATH — the filter is a no-op by design.
            return;
        }

        let projects = detect_scip_projects(root);
        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].dir, root.join("app"));
    }

    #[test]
    fn normalize_language_narrows_supported_set() {
        assert_eq!(normalize_language("rust"), Some("rust"));
        assert_eq!(normalize_language("typescript"), Some("typescript"));
        assert_eq!(normalize_language("csharp"), None);
    }

    #[test]
    fn output_names_root_is_clean_and_subdirs_are_hashed() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let at_root = ScipProject {
            language: "rust",
            dir: root.to_path_buf(),
        };
        let nested = ScipProject {
            language: "typescript",
            dir: root.join("apps/web ui"),
        };
        assert_eq!(output_file_name(&at_root, root), "index-rust.scip");
        let name = output_file_name(&nested, root);
        assert!(
            name.starts_with("index-typescript-apps-web-ui-") && name.ends_with(".scip"),
            "{name}"
        );
    }

    #[test]
    fn lossy_slug_siblings_get_distinct_names() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let dirs = ["apps/web-ui", "apps/web.ui", "apps/web ui", "apps/Web-UI"];
        let names: std::collections::HashSet<String> = dirs
            .iter()
            .map(|d| {
                output_file_name(
                    &ScipProject {
                        language: "typescript",
                        dir: root.join(d),
                    },
                    root,
                )
            })
            .collect();
        assert_eq!(names.len(), dirs.len(), "names collided: {names:?}");
        // Fully non-ASCII dir: slug is empty but the hash still names it.
        let cjk = output_file_name(
            &ScipProject {
                language: "typescript",
                dir: root.join("服务"),
            },
            root,
        );
        assert!(cjk.starts_with("index-typescript-") && cjk.ends_with(".scip"));
    }

    #[test]
    fn expected_output_anchors_at_axil_dir_in_both_modes() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let axil_dir = root.join("brain"); // custom --db layout
        let p = ScipProject {
            language: "rust",
            dir: root.to_path_buf(),
        };
        assert_eq!(
            expected_output(&p, false, &axil_dir, root),
            axil_dir.join("index.scip")
        );
        assert_eq!(
            expected_output(&p, true, &axil_dir, root),
            axil_dir.join("index-rust.scip")
        );
    }
}
