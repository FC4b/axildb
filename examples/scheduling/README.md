# Axil scheduling templates (Phase 12.3)

Run `axil schedule install <name>` to generate these automatically. The files
here are checked-in references for users who prefer hand-rolling.

Supported tasks:

- `daily-brief` — runs `axil brief --window 24h` every morning
- `weekly-retro` — runs `axil retro --window 7d --save` weekly
- `monthly-retro` — runs `axil retro --window 30d --save` monthly

## macOS (launchd)

```bash
axil schedule install daily-brief --hour 8
# Edits ~/Library/LaunchAgents/com.axil.daily-brief.plist
launchctl load ~/Library/LaunchAgents/com.axil.daily-brief.plist
```

See `launchd/com.axil.daily-brief.plist` for the template.

## Linux (systemd --user)

```bash
axil schedule install daily-brief --scheduler systemd --hour 8
# Writes to ~/.axil/schedule/axil-daily-brief.timer and .service templates.
# Copy both to ~/.config/systemd/user/, then:
systemctl --user daemon-reload
systemctl --user enable --now axil-daily-brief.timer
```

See `systemd/axil-daily-brief.timer` + `.service` for templates.

## Cron

```bash
axil schedule install daily-brief --scheduler cron --hour 8
# Writes to ~/.axil/schedule/axil-daily-brief.cron — append to your crontab.
crontab -l > /tmp/crontab.new
cat ~/.axil/schedule/axil-daily-brief.cron >> /tmp/crontab.new
crontab /tmp/crontab.new
```

## Verify

```bash
axil schedule list
axil brief --window 24h        # run the brief by hand to verify output
axil schedule uninstall daily-brief   # remove
```
