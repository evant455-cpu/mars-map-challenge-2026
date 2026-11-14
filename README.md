# Martian EVA Route & Hazard Mapping Engine (Gale Crater)

**NASA Space Apps Challenge 2026 — Martian Map challenge**
Team: Tycho Engineering (solo) · Local Event: Cleveland, MS

A terrain-aware hazard map and route planner for astronaut surface walks (EVAs, extravehicular activities) on Mars. It turns an elevation model into a four-tier hazard map, then uses A* search to plan safety-weighted walking routes between science stops, with walking time, energy and walkback (return-to-base) checks.

> **Project status:** The current demo runs on a **synthetic terrain layout** that imitates features of lower Mount Sharp. Real HiRISE elevation data for the target sites has **not yet been confirmed or loaded**. The code can read a real HiRISE DTM GeoTIFF, but that path has not been tested on real data.

---

## 1. Science targets

The planned EVA visits two features that NASA's Curiosity rover has studied on Mount Sharp (Aeolis Mons) in Gale Crater.

### Primary: Mount Sharp boxwork
- Boxwork is a network of low ridges, roughly 1–2 m tall, with sandy hollows between them, on the slopes of Mount Sharp. [1]
- Scientists think the hardened ridges may have formed from ancient groundwater, and boxwork has not been spotted anywhere else on the mountain. [2]
- The boxwork region is part of a layer of Mount Sharp rich in sulfate minerals, which formed as water was drying out on Mars. [3]
- **Why it suits this tool:** the ridges are natural obstacles and the hollows are natural walking lanes, so the route planner has a real maze to solve.

### Secondary: Gediz Vallis
- Gediz Vallis is a canyon about 9 km long and about 140 m deep. [4]
- Its channel is roughly 91 m wide, with dark sand along its sides and mounds of boulders and debris in its center. [5]
- Curiosity found stones of pure sulfur there, a finding scientists have not yet explained. [6]
- **Why it suits this tool:** steep canyon walls give the planner hard boundaries it must never cross.

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
            • slope (degrees)
            • roughness (3x3 Terrain Ruggedness Index)
            • rock protrusion index (3x3)
            • 1-pixel buffer around missing data
            • 4-tier hazard map + cost grid
                          │
            writes hazard_layers.npz + hazard_manifest.json
                          │
                          ▼
              martian_eva_router.py
            • 8-direction A* search
            • step-to-step slope limit
            • step-by-step walking time
            • walkback check from each stop
                          │
                          ▼
     eva_route.geojson   eva_route_preview.png
```

The mapper is the **single source of truth**: the router reads its hazard layers directly, so the route is always planned on exactly the terrain the hazard map shows.

### Files
| File | What it does |
| :--- | :--- |
| `terrain_hazard_mapper.py` | Builds the elevation model (synthetic, or from a GeoTIFF), computes slope, roughness and rock protrusion, classifies hazards, and saves `hazard_layers.npz`. |
| `martian_eva_router.py` | Loads `hazard_layers.npz`, plans each leg with A*, checks walkback distance, and exports the route. |

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
- **Walking time** is added up step by step using a speed curve inspired by Tobler's hiking function: fastest on a gentle 3° downhill, slower on steep climbs, steep descents and rough ground.
- **Walkback check.** The engine plans a return path from every science stop back to base and warns if any exceeds the limit.

---

## 5. Quickstart

Requirements: Python 3.10+ and `numpy`. Optional: `matplotlib` (preview images) and `rasterio` (reading GeoTIFFs).

```bash
pip install numpy matplotlib
python3 terrain_hazard_mapper.py   # step 1: builds hazard_layers.npz (synthetic layout)
python3 martian_eva_router.py      # step 2: plans the EVA on those layers
```

To use real data, replace `build_synthetic_mount_sharp()` in the mapper's main block with `build_from_geotiff("<your_DTM>.tif")`. Waypoints are set in the router's main block as grid row and column numbers; the first waypoint is treated as base.

### Outputs
| File | Contents |
| :--- | :--- |
| `hazard_layers.npz` | Elevation, slope, roughness, rock protrusion, hazard tier, cost grid and no-go mask |
| `hazard_manifest.json` | Grid size, elevation range, tier coverage and all threshold assumptions |
| `terrain_hazard_map.png` | Shaded relief, slope, and 4-tier hazard map |
| `eva_route.geojson` | Each leg as a 3D line with distance, slope, time and energy |
| `eva_route_preview.png` | The route drawn on the hazard map |
| `hazard_mapper.log`, `eva_router.log` | Run logs |

---

## 6. Demo results (synthetic layout)

Mission: Base → Boxwork stop → Gediz rim stop → Base, on the synthetic 350 m × 450 m layout.

| Leg | Distance | Time | Steepest step | Steps over 20° |
| :--- | ---: | ---: | ---: | ---: |
| Base → Boxwork | 280 m | 8.3 min | 21.5° | 3 (3.2 m) |
| Boxwork → Gediz rim | 151 m | 4.1 min | 20.2° | 1 (1.1 m) |
| Gediz rim → Base | 412 m | 10.3 min | 21.9° | 3 (3.7 m) |
| **Total** | **843 m** | **22.7 min** | | |

- Farthest walkback: 412 m (from the Gediz rim), within the 1,500 m limit.
- No route cell falls on a no-go cell.
- Hazard map coverage: 68% nominal, 8% caution, 9% high-cost, 15% no-go.

These numbers describe the synthetic layout only, not real Mars terrain.

---

## 7. Assumptions and limitations

All of these are planning assumptions, not measured values:

- **Suited astronaut mass:** 180 kg. **Flat-ground walking speed:** 0.85 m/s.
- **Slope limit:** 22°. **Roughness limit:** 0.85 m. **Walkback limit:** 1,500 m.
- **Energy model:** level walking cost (2.8 J per kg per m) plus uphill climb work using Mars gravity (3.721 m/s²) at 25% muscle efficiency. Downhill counts as zero extra effort, so return legs show less energy than they would in reality; braking downhill in a suit does take effort.
- **Science value is chosen by hand.** Stops are picked from published Curiosity findings, not calculated from data.
- **Not modeled:** radiation dose, temperature, dust storms or time-of-day effects.
- **Short steep steps:** routes may use a few steps between 20° and 22° to pass through gaps between ridges instead of making long detours (see the table above).

---

## 8. Sources

1. Space.com, "Curiosity rover finds clues to Mars' watery past in rocky 'spiderwebs.'" https://www.space.com/space-exploration/mars-rovers/curiosity-rover-finds-clues-to-mars-watery-past-in-rocky-spiderwebs
2. NASA, "NASA's Curiosity Mars Rover Starts Unpacking Boxwork Formations." https://www.nasa.gov/missions/mars-science-laboratory/curiosity-rover/nasas-curiosity-mars-rover-starts-unpacking-boxwork-formations
3. NASA press release via Astrobiology.com, "Mars Curiosity Rover Observes Evidence of Past Groundwater" (Feb 23, 2026). https://astrobiology.com/2026/02/mars-curiosity-rover-observes-evidence-of-past-groundwater.html
4. Phys.org / Universe Today, "Curiosity rover is climbing through dramatic striped terrain on Mars" (Mar 4, 2024). https://phys.org/news/2024-03-curiosity-rover-climbing-striped-terrain.pdf
5. NASA/JPL Photojournal, PIA26241, "Curiosity's Route to Gediz Vallis Channel (Rendering)." https://photojournal.jpl.nasa.gov/catalog/PIA26241
6. "Curiosity Rover Discovers Sulphur on Mars as It Leaves Gediz Vallis." https://newsio.com/?p=152730
7. Caltech, "Mars Rover Finds Evidence of Ancient Streambed." https://hr.caltech.edu/news/mars-rover-finds-evidence-ancient-streambed-24666
8. Science News, Nadia Drake, on Curiosity's streambed discovery and Peace Vallis. https://www.sciencenews.org/?p=9013
9. University of Arizona HiRISE, example DTM README (known artifacts section). https://www.uahirise.org/PDS/EXTRAS/DTM/PSP/ORB_003900_003999/PSP_003934_1275_PSP_003855_1275/README.TXT

HiRISE DTM background: NASA PDS Orbital Data Explorer, "MRO HiRISE Digital Terrain Model (DTM)." https://ode.rsl.wustl.edu/mars/pagehelp/Content/Missions_Instruments/Mars%20Reconnaissance%20Orbiter/HiRISE/HiRISE%20DTM.htm

---

## 9. Disclosures

- Background research on Mars terrain was done before the event; all code in this repository was written during the event.
- *(Add any disclosure the Space Apps submission guide requires, such as use of AI assistants.)*
