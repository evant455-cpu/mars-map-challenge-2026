# Martian EVA Route & Hazard Mapping Engine (Gale Crater)

**NASA Space Apps Challenge 2026 — Martian Map challenge**
Team: Southern Astronomy (solo) · Local Event: Cleveland, MS

This is a tool for planning a walk on Mars (an EVA, aka extravehicular activity). You give it elevation data, and it figures out where the ground is too steep or too rough, plans a route between science stops, then checks every step of that walk against the same walkback rule Apollo crews used on the Moon. It all ends up in one interactive map you open in your browser.

> **Project status:** The main demo runs on **real NASA terrain**: a 500 m × 500 m patch of a HiRISE elevation model on the north edge of Mount Sharp. A second, **synthetic** layout is kept for testing. The real patch is real Mount Sharp ground, but I did **not** confirm boxwork or Gediz Vallis inside it.

---

## 1. Science targets

The route planner was designed around two features that NASA's Curiosity rover has studied on Mount Sharp (Aeolis Mons) in Gale Crater.

### Primary: Mount Sharp boxwork
- Boxwork is a network of low ridges, roughly 1–2 m tall, with sandy hollows between them, on the slopes of Mount Sharp. [1]
- Scientists think the hardened ridges may have formed from ancient groundwater, and boxwork has not been spotted anywhere else on the mountain. [2]
- The boxwork region is part of a layer of Mount Sharp rich in sulfate minerals, which formed as water was drying out on Mars. [3]
- **Why I picked it:** the ridges are obstacles and the sandy hollows between them are walking lanes. That turns the route into an actual maze, which is exactly what a route planner needs to prove itself.

### Secondary: Gediz Vallis
- Gediz Vallis is a canyon about 9 km long and about 140 m deep. [4]
- Its channel is roughly 91 m wide, with dark sand along its sides and mounds of boulders and debris in its center. [5]
- Curiosity found stones of pure sulfur there, a finding scientists have not yet explained. [6]
- **Why I picked it:** the canyon walls are too steep to walk, period. The planner has to respect an edge it can NOT cross.

### Sites considered and not chosen
- **Hottah and Link (near Curiosity's landing site):** Curiosity found ancient streambed conglomerate here, the first water-transported gravel identified on Mars. [7] The ground is fairly flat, so it would not exercise the hazard and routing engine much, and it lies far from the Mount Sharp sites.
- **A full Peace Vallis to Mount Sharp traverse:** Peace Vallis alone is a canyon roughly 18 km long. [8] A trip that long needs a multi-day rover expedition, not a single walking EVA.

### What the real data does and doesn't show
- **Used:** HiRISE DTM `DTEEC_001488_1750_001752_1750_U02`, which a 2014 boxwork study also used. [18, 19]
- **Not confirmed:** I could not find clear boxwork in the 500 m patch the demo uses, and this DTM does not reach Gediz Vallis as far as I could tell. The demo is "real north Mount Sharp terrain," not a boxwork-to-Gediz traverse.
- **Not measured:** the real walking distance between a boxwork stop and a Gediz Vallis stop.

---

## 2. Real terrain data

| Item | Value |
| :--- | :--- |
| Product | `DTEEC_001488_1750_001752_1750_U02` ([HiRISE catalog page](https://www.uahirise.org/dtm/PSP_001488_1750)) [18] |
| Source images | HiRISE PSP_001488_1750 and PSP_001752_1750 |
| Producer | USGS Astrogeology (project MSL_TRAVERSE_GAP_1) |
| Format | NASA PDS3 `.IMG`, 32-bit little-endian floats, 360,402,944 bytes |
| Full grid | 6,736 columns × 13,375 rows, 1.0117 m per pixel |
| Full area | 4.7206°S–4.9488°S, 137.2390°E–137.3539°E |
| **Demo patch** | rows 4550–5049, columns 4850–5349 (500 × 500 px, about 506 m × 506 m) |
| Patch corner (top-left) | 4.7982°S, 137.3217°E |
| Patch elevation | −4,040.5 m to −3,943.4 m (about 97 m of relief), no missing cells |

**Download it** (about 25 seconds). The file is too big for GitHub, so `realdata/` is in `.gitignore`:

```bash
mkdir realdata && cd realdata
B=https://www.uahirise.org/PDS
curl -o DTEEC_001488_1750_001752_1750_U02.IMG $B/DTM/PSP/ORB_001400_001499/PSP_001488_1750_PSP_001752_1750/DTEEC_001488_1750_001752_1750_U02.IMG
curl -o README.TXT $B/EXTRAS/DTM/PSP/ORB_001400_001499/PSP_001488_1750_PSP_001752_1750/README.TXT
cd ..
```

**How it's read:** `pds_dtm_loader.py` memory-maps the file (`np.memmap`), so only the 500 × 500 patch is loaded, not all 360 MB. It skips the 26,944-byte text label, reads `<f4` floats, turns the "no data" marker (about −3.4 × 10³⁸) into `NaN`, and refuses any patch that is more than half missing data. Pixel positions convert to latitude and longitude with the equirectangular projection values from the file's own label.

**Known artifacts:** the far west strip of this DTM shows stitching seams, so the loader warns about patches there. The HiRISE readme also warns about square "box" artifacts 0.5–1 m off from their surroundings [9]; `--despike-seams` applies a median filter for those.

---

## 3. How it works

```text
 [ HiRISE DTM .IMG ]                     [ Synthetic Mount Sharp layout ]
        │                                              │
 pds_dtm_loader.py + run_mapper_real.py    terrain_hazard_mapper.py (main)
        │                                              │
        └──────────────► terrain_hazard_mapper ◄───────┘
                         • slope, roughness, rock protrusion
                         • 1-pixel buffer around missing data
                         • 4-tier hazard map + cost grid
                                    │  hazard_layers.npz
                                    ▼
                         martian_eva_router.py
                         • 8-direction A* search
                         • step-to-step slope limit
                         • walking time, energy, walkback distance
                                    │  eva_route.geojson
                                    ▼
                         astronaut_safety_layer.py
                         • time-to-safety map, Apollo walkback check
                         • timeline, radiation, line of sight, rest stops
                                    │  safety_layers.npz + safety_report.json
                                    ▼
                         build_web_viewer.py → eva_map_viewer.html
```

The hazard mapper is the **single source of truth**: every later step reads its layers, so the route, the safety checks and the map all describe exactly the same terrain.

### Files
| File | What it does |
| :--- | :--- |
| `pds_dtm_loader.py` | Reads a patch of the HiRISE PDS3 file, masks missing data, converts pixels to latitude/longitude. |
| `run_mapper_real.py` | Runs the hazard mapper on the real patch with the true 1.0117 m pixel size. |
| `terrain_hazard_mapper.py` | Computes slope, roughness and rock protrusion, classifies hazards, saves `hazard_layers.npz`. Its own main builds the synthetic layout. |
| `martian_eva_router.py` | Plans each leg with A*, checks walkback distance, exports the route. |
| `astronaut_safety_layer.py` | Builds the time-to-safety map and checks the route against the walkback rule, line of sight and climbing effort. |
| `build_web_viewer.py` | Packs everything into one interactive, offline HTML page. The title and badge switch automatically between "HiRISE DTM" and "Synthetic Layout." |

---

## 4. Hazard classification

Each grid cell gets three measurements:

1. **Slope (degrees)**, from neighboring elevations (`np.gradient`).
2. **Roughness (m):** the Terrain Ruggedness Index, the root-mean-square height difference between a cell and its 8 neighbors.
3. **Rock protrusion index (m):** how far a cell rises above the average of its 8 neighbors. Wide boulders raise the neighboring cells too, so this measures local crest prominence, **not** true rock height.

| Tier | Meaning | Triggered when (assumed thresholds) |
| :---: | :--- | :--- |
| 0 | Nominal | none of the triggers below |
| 1 | Caution | slope > 10° or roughness > 0.25 m |
| 2 | High-cost | slope > 15°, roughness > 0.50 m, or protrusion > 0.30 m |
| 3 | No-go | slope > 22°, roughness > 0.85 m, protrusion > 0.65 m, or within 1 pixel of missing data |

Passable tiers get a cost multiplier that grows with severity (1.0x for nominal; roughly 1.6–2.5x caution and 4.4–9.8x high-cost on the synthetic layout); no-go cells are blocked.

---

## 5. Route planning

- **A\* search, 8 directions.** The heuristic (the estimate of remaining cost) is straight-line grid distance at the cheapest possible rate, so it never overestimates and A* returns the lowest-cost path.
- **Cost of each step** = surface distance × the destination cell's cost multiplier, plus a penalty per meter of climb.
- **Step slope limit.** Besides the cell's own slope, the slope between each pair of neighboring cells must stay at or below 22°.
- **Walking time** is added up step by step using a speed curve inspired by Tobler's hiking function: fastest on a gentle 3° downhill, slower on steep climbs, steep descents and rough ground. The flat-ground speed, 0.85 m/s (about 3.1 km/h), is close to the 3.3 km/h walking return speed Apollo used in its planning. [10]

---

## 6. Astronaut safety layer

The safety checks are modeled on how Apollo crews worked on the Moon and on what Curiosity has measured at Gale Crater.

### Lessons from Apollo
- **Always be able to walk home.** If the rover broke down at any point, the crew had to be able to walk back to the lunar module, so how far they could go was capped the whole time. [11] That cap came from how much life support was left in the backpack, with 30 minutes held back as a reserve. [10]
- **Uphill is what wears you out.** On Apollo 14, Shepard and Mitchell climbed toward the rim of Cone Crater on a 1.4 km walk. Coming back downhill, they doubled their average speed. On Apollo 12, a crew had to stop and rest after only 300 m up a shallow 3% grade. [12] Sources don't agree on the Apollo 14 slope: about 10% [13] or ten degrees [12], so somewhere around 6–10°.
- **Knowing where you are counts too.** Without landmarks, the Apollo 14 crew couldn't judge distances. Running short on time and oxygen, they turned back about 30 m from the rim and never knew how close they were. [14]

### What the layer computes
1. **Time-to-safety map.** One reverse Dijkstra search, starting at base, records the fastest walking time back to base from every cell, using the same speed curve and step slope limit as the router. It appears in the viewer as the "Time to Safety" layer.
2. **Apollo walkback check at every route point:**
   `time used so far + time back to base + reserve ≤ EVA time budget`
   Points that fail are drawn in red.
3. **Timeline and radiation.** Curiosity's Radiation Assessment Detector measured an average dose of 0.67 mSv per day on the surface from August 2012 to June 2013. [15] EVA dose = hours outside × 0.67 / 24. The bigger risk is a sudden solar particle event [16], so the layer also reports the **worst-case time to shelter** along the route.
4. **Line of sight to base**, from a 1.7 m eye height to a 3.0 m mast at base. Points where terrain blocks the view are drawn dashed.
5. **Sustained-climb rest stops**, where the route climbs more than 5 m within 50 m of walking.
6. **Temperature guidance (not modeled spatially).** Curiosity measured daily air highs near 0°C and lows near −70°C over late winter through spring [17], so EVAs are planned around midday.

---

## 7. Quickstart

Requirements: Python 3.10+ and `numpy`. Optional: `matplotlib` for the preview images. The real data needs no extra GIS libraries.

```bash
pip install numpy matplotlib
```

### Real HiRISE terrain (main demo)

Download the DTM first (Section 2), then, in the project folder:

```bash
python3 run_mapper_real.py          # 1. hazard layers from the real 500 m patch
python3 martian_eva_router.py       # 2. plan the EVA route
python3 astronaut_safety_layer.py   # 3. safety checks
python3 build_web_viewer.py         # 4. interactive map
```

Open `eva_map_viewer.html` in a desktop browser. It works offline; touch panning also works on tablets.

To try another patch: `python3 run_mapper_real.py --row 8550 --col 3350`. You'll also need new waypoints for it.

### Synthetic layout (test terrain)

Run `python3 terrain_hazard_mapper.py` instead of step 1. The router's main block currently holds the **real-patch** waypoints, so swap in the synthetic ones first: Base (row 45, col 60), Boxwork Vein Node (195, 180), Gediz Sulfur Rim (255, 298), Base again.

### Waypoints on the real patch

Row and column are positions inside the 500 × 500 patch. Each was chosen as the flattest tier-0 spot in the patch's largest connected walkable area, which covers 81% of it.

| Waypoint | Row | Col |
| :--- | ---: | ---: |
| HAB / Base | 169 | 361 |
| Stop A (center) | 319 | 224 |
| Stop B (southeast) | 393 | 449 |

Safety assumptions can be changed from the command line, for example `python3 astronaut_safety_layer.py --eva-budget-min 90 --reserve-min 30`.

### Outputs
| File | Contents |
| :--- | :--- |
| `hazard_layers.npz` | Elevation, slope, roughness, rock protrusion, hazard tier, cost grid and no-go mask |
| `hazard_manifest.json` | Source, grid size, elevation range, tier coverage and all threshold assumptions |
| `terrain_hazard_map.png` | Shaded relief, slope, and 4-tier hazard map |
| `eva_route.geojson` | Each leg as a 3D line with distance, slope, time and energy |
| `eva_route_preview.png` | The route drawn on the hazard map |
| `safety_layers.npz` | Time-to-safety grid and per-point safety flags along the route |
| `safety_report.json` | Timeline, walkback result, radiation, shelter time, line of sight, rest stops |
| `astronaut_safety_preview.png` | Time-to-safety map with route safety flags |
| `eva_map_viewer.html` | The interactive map |

---

## 8. Results

### Real HiRISE patch (north Mount Sharp)

Mission: Base → Stop A → Stop B → Base. Hazard map: 49% nominal, 19% caution, 15% high-cost, 17% no-go.

| Leg | Distance | Climb | Max slope | Time |
| :--- | ---: | ---: | ---: | ---: |
| Base → Stop A | 224.0 m | +21.9 m | 17.1° | 7.4 min |
| Stop A → Stop B | 418.6 m | +9.9 m | 16.1° | 11.2 min |
| Stop B → Base | 270.4 m | +2.1 m | 9.9° | 6.3 min |
| **Total** | **913.1 m** | | | **24.9 min** |

| Safety check | Result |
| :--- | :--- |
| Walkback rule (60 min budget, 30 min reserve) | Pass at every route point; smallest margin 5.1 min |
| Worst-case time to shelter | 6.2 min (at Stop B) |
| Radiation dose for the walk | 0.012 mSv |
| Line of sight to base | 64.8% of route points |
| Sustained-climb rest stops | 2 (largest climb +16.3 m) |
| Steps over 20° | 0 |

### Synthetic layout

Mission: Base → Boxwork stop → Gediz rim stop → Base. Hazard map: 68% nominal, 8% caution, 9% high-cost, 15% no-go.

| Leg | Distance | Time | Steps over 20° |
| :--- | ---: | ---: | ---: |
| Base → Boxwork | 280 m | 8.3 min | 3 |
| Boxwork → Gediz rim | 151 m | 4.1 min | 1 |
| Gediz rim → Base | 412 m | 10.3 min | 3 |
| **Total** | **843 m** | **22.7 min** | |

Walkback pass (smallest margin 7.3 min), worst time to shelter 9.0 min, line of sight 99.7%, 0 rest stops.

### What the comparison shows

Real Mars is rougher than my synthetic version: 49% safe ground versus 68%. On the real patch, 35% of the walk is out of sight of base and 2 rest stops show up, where the synthetic layout had almost none. That's the safety layer doing its job on actual terrain.

One more thing to know: time to shelter can be shorter than the planned walk home. The safety map finds the *fastest* way back, while the router plans the *safest* route, which avoids high-cost ground.

---

## 9. Assumptions and limitations

All of these are planning assumptions, not measured values:

- **Suited astronaut mass:** 180 kg. **Flat-ground walking speed:** 0.85 m/s.
- **Slope limit:** 22°. **Roughness limit:** 0.85 m. **Walkback distance limit:** 1,500 m.
- **EVA time budget:** 60 min with a 30 min reserve. Apollo's reserve inspired the 30 minutes; the 60-minute budget is a choice for this demo, not a NASA figure for a Mars suit.
- **Timeline counts walking only.** Time spent working at each science stop is not included yet.
- **Energy model:** level walking cost (2.8 J per kg per m) plus uphill climb work using Mars gravity (3.721 m/s²) at 25% muscle efficiency. Downhill counts as zero extra effort, so return legs show less energy than they would in reality.
- **Radiation:** 0.67 mSv/day is a 2012–2013 average and varies with the solar cycle. Shielding by terrain or the suit is not modeled.
- **Line of sight** is geometric only; it does not model radio range.
- **Waypoints on the real patch** were chosen for safe, flat ground, not for science value. Stops on the synthetic layout were picked by hand from published Curiosity findings.
- **DTM accuracy:** HiRISE DTMs can contain artifacts [9]; the demo patch was not despiked.
- **Not modeled:** dust storms, wind, or how temperature changes along the route.

---

## 10. Sources

1. Space.com, "Curiosity rover finds clues to Mars' watery past in rocky 'spiderwebs.'" https://www.space.com/space-exploration/mars-rovers/curiosity-rover-finds-clues-to-mars-watery-past-in-rocky-spiderwebs
2. NASA, "NASA's Curiosity Mars Rover Starts Unpacking Boxwork Formations." https://www.nasa.gov/missions/mars-science-laboratory/curiosity-rover/nasas-curiosity-mars-rover-starts-unpacking-boxwork-formations
3. NASA press release via Astrobiology.com, "Mars Curiosity Rover Observes Evidence of Past Groundwater" (Feb 23, 2026). https://astrobiology.com/2026/02/mars-curiosity-rover-observes-evidence-of-past-groundwater.html
4. Phys.org / Universe Today, "Curiosity rover is climbing through dramatic striped terrain on Mars" (Mar 4, 2024). https://phys.org/news/2024-03-curiosity-rover-climbing-striped-terrain.pdf
5. NASA/JPL Photojournal, PIA26241, "Curiosity's Route to Gediz Vallis Channel (Rendering)." https://photojournal.jpl.nasa.gov/catalog/PIA26241
6. "Curiosity Rover Discovers Sulphur on Mars as It Leaves Gediz Vallis." https://newsio.com/?p=152730
7. Caltech, "Mars Rover Finds Evidence of Ancient Streambed." https://hr.caltech.edu/news/mars-rover-finds-evidence-ancient-streambed-24666
8. Science News, Nadia Drake, on Curiosity's streambed discovery and Peace Vallis. https://www.sciencenews.org/?p=9013
9. University of Arizona HiRISE, example DTM README (known artifacts section). https://www.uahirise.org/PDS/EXTRAS/DTM/PSP/ORB_003900_003999/PSP_003934_1275_PSP_003855_1275/README.TXT
10. NASA, Apollo traverse planning document on life-support consumables, walkback and rideback limits (PDF). https://www.nasa.gov/wp-content/uploads/static/history/alsj/19790073512_1979073512.pdf
11. "Experiment Operations During Apollo EVAs: Lunar Rover Vehicle." https://www.ninfinger.org/karld/My%20Space%20Museum/LRV.htm
12. OSTI, report on Apollo EVA experience and lunar base surface operations (PDF). https://www.osti.gov/servlets/purl/7168757
13. Apollo Lunar Surface Journal, "Apollo 14 Summary." https://astro.if.ufrgs.br/solar/apo14.htm
14. Universe Today, "Latest LRO Image Solves Apollo 14 Mystery." https://www.universetoday.com/37870/latest-lro-image-solves-apollo-14-mystery/amp
15. ScienceDaily / Southwest Research Institute, "First radiation measurements from the surface of Mars" (Dec 9, 2013). https://www.sciencedaily.com/releases/2013/12/131209181133.htm
16. redOrbit, "Mars Surface Radiation Levels Almost Suitable For Manned Mission." https://www.redorbit.com/news/space/1113024596/mars-surface-radiation-almost-suitable-for-manned-mission-121013
17. NASA, "Steady Temperatures at Mars' Gale Crater." https://science.nasa.gov/resource/steady-temperatures-at-mars-gale-crater
18. University of Arizona HiRISE, DTM "Edge along Gale Crater Interior Mound" (DTEEC_001488_1750_001752_1750_U02). https://www.uahirise.org/dtm/PSP_001488_1750
19. Siebach et al. (2014), "Volumetric estimates of ancient water on Mount Sharp based on boxwork deposits, Gale Crater, Mars," Journal of Geophysical Research: Planets. https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2013JE004508

**Data credit:** HiRISE images NASA/JPL/University of Arizona; DTM produced by USGS Astrogeology.

---

## 11. Disclosures

- Background research on Mars terrain was done before the event; all code in this repository was written during the event.
- **AI assistance.** I built this project solo with heavy help from AI assistants, and I want to be upfront about how:
    - **Claude (Anthropic)** helped me plan the weekend, scope the MVP, and research the science and Apollo sources. It reviewed and tested my code by running it, found bugs, and wrote fixes (the map viewer's label sizing, zoom and touch panning, the loader's logging and missing-data check, and the data-driven map title). It also found and downloaded the HiRISE DTM, picked the 500 m patch and the waypoints, and drafted most of this README.
    - **A second AI assistant** generated much of the Python code from my descriptions and requirements, including the hazard mapper, router, safety layer and viewer.
    - **My part:** I chose the challenge, the science targets and the safety ideas to model, made the design calls, directed every change, compared the two assistants' advice against each other, and decided what to keep. Every number in the Results section comes from running the code, not from an AI's guess.
