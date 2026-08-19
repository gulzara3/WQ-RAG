# Knowledge base — 25 documents in four categories (Section 2.2.3)

Drop the files into the sub-folders below and run `python scripts/04_build_knowledge_base.py`.
PDF/TXT/MD/HTML are ingested; files beginning with `_`, `README` or `MANIFEST` are ignored.
Copyrighted PDFs are **not** redistributed in this repository — obtain them from the links.
The station-metadata `.txt` files and the thresholds reference are authored for this study
and are included.

## 1. `regulations/` — Regulatory standards (3)

| # | Document | Source |
|---|----------|--------|
| 1 | US EPA National Primary Drinking Water Regulations (40 CFR Part 141) | https://www.epa.gov/ground-water-and-drinking-water/national-primary-drinking-water-regulations · https://www.ecfr.gov/current/title-40/chapter-I/subchapter-D/part-141 |
| 2 | US EPA Secondary Drinking Water Standards (40 CFR Part 143) | https://www.epa.gov/sdwa/secondary-drinking-water-standards-guidance-nuisance-chemicals |
| 3 | WHO Guidelines for Drinking-water Quality, 4th ed. incl. addenda (2022) | https://www.who.int/publications/i/item/9789240045064 |

## 2. `technical_guides/` — Technical guides (7)

| # | Document | Source |
|---|----------|--------|
| 4 | USGS Techniques & Methods 1-D3 — Continuous water-quality monitors (Wagner et al., 2006) | https://pubs.usgs.gov/tm/2006/tm1D3/pdf/TM1D3.pdf |
| 5 | USGS parameter-code reference (codes 00010, 00095, 00300, 00400, 63680) | https://help.waterdata.usgs.gov/codes-and-parameters/parameters |
| 6 | EPA 2018 Drinking Water Standards and Health Advisories Table | https://www.epa.gov/sdwa/2018-drinking-water-standards-and-advisory-tables |
| 7 | EPA drinking-water monitoring & compliance guidance (2013) | https://www.epa.gov/dwreginfo |
| 8 | EPA Aquatic Life Criteria (Gold Book) — DO, pH, temperature | https://www.epa.gov/wqc/national-recommended-water-quality-criteria-aquatic-life-criteria-table |
| 9 | Sensor calibration / QA protocols (from TM 1-D3 chapters) | as #4 |
| 10 | `Enhanced_WQ_Thresholds.txt` — consolidated per-parameter thresholds + station baselines | **included** |

## 3. `case_studies/` — 10 peer-reviewed studies (11–20)

Bonet et al. 2026 (Water 18:403); El-Shafeiy et al. 2023; Hou et al. 2013; Leigh et al. 2019;
Liao et al. 2024; Liu et al. 2020; Mao et al. 2017; Perelman et al. 2012;
Santos-Fernandez et al. 2024; Wang et al. 2023.  Full references are in the manuscript bibliography.
Save each as `<FirstAuthor>_<Year>.pdf` (or a `.txt` abstract+findings summary if the PDF licence forbids storage).

## 4. `station_metadata/` — Station context (5)

| # | File | Content |
|---|------|---------|
| 21 | `station_01646500_potomac.txt` | Potomac River, DC — urban, combined-sewer overflows; DC DCMR Title 21 |
| 22 | `station_03351000_white.txt` | White River, IN — row-crop agriculture on glacial till; 327 IAC 2-1 |
| 23 | `station_14211010_clackamas.txt` | Clackamas River, OR — volcanic source water, 2020 Riverside Fire; OAR 340-041 |
| 24 | `station_11447650_sacramento.txt` | Sacramento River, CA — tidal/agricultural; CVRWQCB Basin Plan, NMFS winter-run Chinook |
| 25 | `regulatory_jurisdictions.txt` | Cross-walk of federal/state/species criteria applied per station |

State regulation sources: Oregon OAR 340-041 https://secure.sos.state.or.us/oard/displayDivisionRules.action?selectedDivision=1458 ·
California CVRWQCB Basin Plan https://www.waterboards.ca.gov/centralvalley/water_issues/basin_plans/ ·
DC DCMR Title 21 https://www.dcregs.dc.gov/Common/DCMR/RuleList.aspx?TitleNum=21 ·
Indiana 327 IAC 2 https://www.in.gov/legislative/iac/iac_title?iact=327&iaca=2
