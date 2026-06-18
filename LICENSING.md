# Licensing

Axil is **free for noncommercial use** and **available under a commercial license**
for everything else. This is a dual-licensing model.

## TL;DR

| You are…                                                        | License you use                       | Cost |
|-----------------------------------------------------------------|---------------------------------------|------|
| An individual hacking on a hobby/personal project               | PolyForm Noncommercial 1.0.0          | Free |
| A researcher, student, or nonprofit / educational / gov. org    | PolyForm Noncommercial 1.0.0          | Free |
| Evaluating Axil, running tests, or building a proof-of-concept  | PolyForm Noncommercial 1.0.0          | Free |
| A company using Axil in or for a commercial product/service     | **Commercial License**                | Paid |

If you are unsure which applies, assume you need a commercial license and
[get in touch](#getting-a-commercial-license).

## Noncommercial use — free

The source in this repository is licensed under the
[PolyForm Noncommercial License 1.0.0](./LICENSE). In plain terms, you may use,
modify, and redistribute Axil for **any noncommercial purpose** at no cost,
including:

- Personal study, hobby projects, experimentation, and research.
- Use by charities, educational institutions, public research organizations,
  public safety/health organizations, and government institutions.

"Noncommercial" is defined by the license text, not by this summary — read
[`LICENSE`](./LICENSE) for the controlling terms.

> **Note:** PolyForm Noncommercial is a *source-available* license, **not** an
> OSI-approved open-source license. Axil is source-available, not open source.

## Commercial use — license required

You need a commercial license if you (or your company) use Axil **in connection
with a commercial product or service** — for example embedding Axil in a product
you sell, in an internal tool at a for-profit company, or in a paid/hosted
service. The commercial license removes the noncommercial restriction and can
include support and indemnification terms.

### Getting a commercial license

Contact: **seksan.dev@gmail.com** (subject: "Axil commercial license").

## Axil Atlas is a separate, commercial product

**Axil Atlas** — the multi-database control plane (cross-machine distillate sync,
team memory compounding, managed/self-hosted server) — is **not** part of this
repository and is **not** covered by this license. Atlas is closed-source and
offered under its own commercial terms. This repository contains **no Atlas
code**: the only integration point is the generic `CanonicalPublisher` trait in
`axil-core`, which any external coordinator may implement. Axil is a complete,
standalone local memory engine with zero Atlas dependency.

## Contributions

By contributing to this repository you agree that your contributions are
licensed under the same terms and that the maintainers may also offer your
contributions under the commercial license (dual-licensing requires this). A
formal Contributor License Agreement (CLA) may be introduced before the repo is
made public.

## Trademarks

"Axil" and the Axil logo are trademarks of the project owner. The source license
does **not** grant trademark rights — a fork may use the code under the
applicable license but may not use the Axil name or branding to market a
competing product.
