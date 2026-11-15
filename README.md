# Martian EVA Route & Hazard Mapping Engine (Gale Crater)

**NASA Space Apps Challenge 2026 — Martian Map challenge**
Team: Tycho Engineering (solo) · Local Event: Cleveland, MS

This is a tool for planning a walk on Mars (an EVA, aka extravehicular activity). You give it elevation data, and it figures out where the ground is too steep or too rough, plans a route between science stops, then checks every step of that walk against the same walkback rule Apollo crews used on the Moon. It all ends up in one interactive map you open in your browser.

> **Project status:** The current demo runs on a **synthetic terrain layout** that imitates features of lower Mount Sharp. Real HiRISE elevation data for the target sites has **not yet been confirmed or loaded**. The code can read a real HiRISE DTM GeoTIFF, but that path has not been tested on real data.

---

## 1. Science targets

The planned EVA visits two features that NASA's Curiosity rover has studied on Mount Sharp (Aeolis Mons) in Gale Crater.

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

### Not yet verified
- The real distance between a boxwork stop and a Gediz Vallis stop. In the synthetic layout they are about 250 m apart; the real spacing must be measured on a real elevation model, and the two sites may be too far apart for one EVA.
- Which HiRISE DTM products, if any, cover these sites.

---

## 2. How it works

```text
 [ HiRISE DTM GeoTIFF (.tif) ]  OR  [ Synthetic Mount Sharp layout ]
                          │
                          ▼
            terrain_hazard_mapper.py
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
            • time-to-safety map (walking minutes to base)
            • Apollo walkback check at every route point
            • timeline, radiation dose, line of sight,
              sustained-climb rest stops
                          │  safety_layers.npz + safety_report.json
                          ▼
              build_web_viewer.py
            • one offline HTML map: 5 layers, route,
              safety overlays, timeline panel, profile
                          │
                          ▼
                  eva_map_viewer.html
```

The hazard mapper is the **single source of truth**: every later step reads its layers, so the route, the safety checks and the map all describe exactly the same terrain.

### Files
| File | What it does |
| :--- | :--- |
| `terrain_hazard_mapper.py` | Builds the elevation model (synthetic, or from a GeoTIFF), computes slope, roughness and rock protrusion, classifies hazards, and saves `hazard_layers.npz`. |
| `martian_eva_router.py` | Loads the hazard layers, plans each leg with A*, checks walkback distance, and exports the route. |
| `astronaut_safety_layer.py` | Builds the time-to-safety map and checks the route against the walkback rule, line of sight and climbing effort. |
| `build_web_viewer.py` | Packs everything into one interactive, offline HTML page. |

---

## 3. Hazard classification

Each 1 m grid cell gets three measurements:

1. **Slope (degrees)**, from neighboring elevations (`np.gradient`).
2. **Roughness (m):** the Terrain Ruggedness Index, the root-mean-square height difference between a cell and its 8 neighbors.
3. **Rock protrusion index (m):** how far a cell rises above the average of its 8 neighbors. Wide boulders raise the neighboring cells too, so this measures local crest prominence, **not** true rock height.

| Tier | Meaning | Triggered when (assumed thresholds) | Cost multiplier* |
| :---: | :--- | :--- | :---: |
| 0 | Nominal | none of the triggers below | 1.0x |
| 1 | Caution | slope > 10° or roughness > 0.25 m | 1.6x – 2.5x |
| 2 | High-cost | slope > 15°, roughness > 0.50 m, or protrusion > 0.30 m | 4.4x – 9.8x |
| 3 | No-go | slope > 22°, roughness > 0.85 m, protrusion > 0.65 m, or within 1 pixel of missing data | blocked |

\*Cost ranges measured on the synthetic layout; they depend on the terrain.

**Optional despiking:** setting `despike_hirise_seams=True` applies a 3x3 median filter before analysis. Some HiRISE DTMs contain square areas about 0.5–1 m off from their surroundings [9]; that is close to the roughness limits above, so such artifacts could otherwise appear as false obstacles.

---

## 4. Route planning

- **A\* search, 8 directions.** The heuristic (the estimate of remaining cost) is straight-line grid distance at the cheapest possible rate, so it never overestimates and A* returns the lowest-cost path.
- **Cost of each step** = surface distance × the destination cell's cost multiplier, plus a penalty per meter of climb.
- **Step slope limit.** Besides the cell's own slope, the slope between each pair of neighboring cells must stay at or below 22°.
- **Walking time** is added up step by step using a speed curve inspired by Tobler's hiking function: fastest on a gentle 3° downhill, slower on steep climbs, steep descents and rough ground. The flat-ground speed, 0.85 m/s (about 3.1 km/h), is close to the 3.3 km/h walking return speed Apollo used in its planning. [10]

---

## 5. Astronaut safety layer

The safety checks are modeled on how Apollo crews worked on the Moon and on what Curiosity has measured at Gale Crater.

### Lessons from Apollo
- **Always be able to walk home.** If the rover broke down at any point, the crew had to be able to walk back to the lunar module, so how far they could go was capped the whole time. [11] That cap came from how much life support was left in the backpack, with 30 minutes held back as a reserve. [10]
- **Uphill is what wears you out.** On Apollo 14, Shepard and Mitchell climbed toward the rim of Cone Crater on a 1.4 km walk. Coming back downhill, they doubled their average speed. On Apollo 12, a crew had to stop and rest after only 300 m up a shallow 3% grade. [12] Sources don't agree on the Apollo 14 slope: about 10% [13] or ten degrees [12], so somewhere around 6–10°.
- **Knowing where you are counts too.** Without landmarks, the Apollo 14 crew couldn't judge distances. Running short on time and oxygen, they turned back about 30 m from the rim and never knew how close they were. [14]

### What the layer computes
1. **Time-to-safety map.** One reverse Dijkstra search, starting at base, records the fastest walking time back to base from every cell on the map, using the same speed curve and step slope limit as the router. It appears in the viewer as the "Time to Safety" layer.
2. **Apollo walkback check at every route point:**
   `time used so far + time back to base + reserve ≤ EVA time budget`
   Points that fail are drawn in red.
3. **Timeline and radiation.** Curiosity's Radiation Assessment Detector measured an average dose of 0.67 mSv per day on the surface from August 2012 to June 2013. [15] EVA dose = hours outside × 0.67 / 24. The bigger risk is a sudden solar particle event [16], so the layer also reports the **worst-case time to shelter** along the route.
4. **Line of sight to base**, from a 1.7 m eye height to a 3.0 m mast at base. Points where terrain blocks the view are drawn dashed.
5. **Sustained-climb rest stops**, where the route climbs more than 5 m within 50 m of walking.
6. **Temperature guidance (not modeled spatially).** Curiosity measured daily air highs near 0°C and lows near −70°C over late winter through spring [17], so EVAs are planned around midday.

---

## 6. Quickstart

Requirements: Python 3.10+ and `numpy`. Optional: `matplotlib` (preview images) and `rasterio` (reading GeoTIFFs).

```bash
pip install numpy matplotlib
python3 terrain_hazard_mapper.py    # 1. hazard layers (synthetic layout)
python3 martian_eva_router.py       # 2. plan the EVA route
python3 astronaut_safety_layer.py   # 3. safety checks
python3 build_web_viewer.py         # 4. interactive map
```

Then open `eva_map_viewer.html` in a desktop browser. It works offline; touch panning also works on tablets.

To use real data, replace `build_synthetic_mount_sharp()` in the mapper's main block with `build_from_geotiff("<your_DTM>.tif")`. Waypoints are set in the router's main block as grid row and column numbers; the first waypoint is treated as base. Safety assumptions can be changed with command-line options, for example `python3 astronaut_safety_layer.py --eva-budget-min 90 --reserve-min 30`.

### Outputs
| File | Contents |
| :--- | :--- |
| `hazard_layers.npz` | Elevation, slope, roughness, rock protrusion, hazard tier, cost grid and no-go mask |
| `hazard_manifest.json` | Grid size, elevation range, tier coverage and all threshold assumptions |
| `terrain_hazard_map.png` | Shaded relief, slope, and 4-tier hazard map |
| `eva_route.geojson` | Each leg as a 3D line with distance, slope, time and energy |
| `eva_route_preview.png` | The route drawn on the hazard map |
| `safety_layers.npz` | Time-to-safety grid and per-point safety flags along the route |
| `safety_report.json` | Timeline, walkback result, radiation, shelter time, line of sight, rest stops |
| `astronaut_safety_preview.png` | Time-to-safety map with route safety flags |
| `eva_map_viewer.html` | The interactive map |
| `*.log` | Run logs |

---

## 7. Demo results (synthetic layout)

Mission: Base → Boxwork stop → Gediz rim stop → Base, on the synthetic 350 m × 450 m layout.

| Leg | Distance | Time | Steepest step | Steps over 20° |
| :--- | ---: | ---: | ---: | ---: |
| Base → Boxwork | 280 m | 8.3 min | 21.5° | 3 (3.2 m) |
| Boxwork → Gediz rim | 151 m | 4.1 min | 20.2° | 1 (1.1 m) |
| Gediz rim → Base | 412 m | 10.3 min | 21.9° | 3 (3.7 m) |
| **Total** | **843 m** | **22.7 min** | | |

| Safety check | Result |
| :--- | :--- |
| Walkback rule (60 min budget, 30 min reserve) | Pass at all 720 route points; smallest margin 7.3 min |
| Worst-case time to shelter | 9.0 min (at the Gediz rim) |
| Radiation dose for the walk | 0.011 mSv |
| Line of sight to base | 99.7% of route points |
| Sustained-climb rest stops | 0 (no stretch climbs 5 m within 50 m) |
| EVA window | 12:00–12:23 LMST, walking time only |

- No route cell falls on a no-go cell.
- Hazard map coverage: 68% nominal, 8% caution, 9% high-cost, 15% no-go.
- The time to shelter from the Gediz rim (9.0 min) is shorter than the planned return leg (10.3 min). That is expected: the safety map finds the *fastest* walk home, while the router plans the *safest* route, which avoids high-cost ground.

These numbers describe the synthetic layout only, not real Mars terrain.

---

## 8. Assumptions and limitations

All of these are planning assumptions, not measured values:

- **Suited astronaut mass:** 180 kg. **Flat-ground walking speed:** 0.85 m/s.
- **Slope limit:** 22°. **Roughness limit:** 0.85 m. **Walkback distance limit:** 1,500 m.
- **EVA time budget:** 60 min with a 30 min reserve. Apollo's reserve inspired the 30 minutes; the 60-minute budget is a choice for this demo, not a NASA figure for a Mars suit.
- **Timeline counts walking only.** Time spent working at each science stop is not included yet.
- **Energy model:** level walking cost (2.8 J per kg per m) plus uphill climb work using Mars gravity (3.721 m/s²) at 25% muscle efficiency. Downhill counts as zero extra effort, so return legs show less energy than they would in reality.
- **Radiation:** 0.67 mSv/day is a 2012–2013 average and varies with the solar cycle. Shielding by terrain or the suit is not modeled.
- **Line of sight** is geometric only; it does not model radio range.
- **Science value is chosen by hand.** Stops are picked from published Curiosity findings, not calculated from data.
- **Not modeled:** dust storms, wind, or how temperature changes along the route.
- **Short steep steps:** routes may use a few steps between 20° and 22° to pass through gaps between ridges instead of making long detours.

---

## 9. Sources

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

HiRISE DTM background: NASA PDS Orbital Data Explorer, "MRO HiRISE Digital Terrain Model (DTM)." https://ode.rsl.wustl.edu/mars/pagehelp/Content/Missions_Instruments/Mars%20Reconnaissance%20Orbiter/HiRISE/HiRISE%20DTM.htm

---

## 10. Disclosures

- Background research on Mars terrain was done before the event; all code in this repository was written during the event.
- *(Add any disclosure the Space Apps submission guide requires, such as use of AI assistants.)*
