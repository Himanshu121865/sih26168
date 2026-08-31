#!/usr/bin/env python3
"""
build_graph.py — Step 10.2 — OSM PBF -> road graph for on-device HMM matching.

Extracts highway ways from a Geofabrik PBF and exports a compact node/edge JSON
consumed by android app assets (maps/road_graph.json) and python/hmm_matcher.py.

Usage:
  osmium tags-filter maps/city.osm.pbf w/highway -o maps/highway.osm.pbf
  python python/maps/build_graph.py --pbf maps/highway.osm.pbf --out python/hmm/road_graph.json

Output format: [ {"lat": [...], "lon": [...]}, [[node_a, node_b], ...] ]
(node index arrays; ~5MB for a city, loads instantly on device)
"""
import argparse, json
from pathlib import Path

# Requires `esythe`/`osmium` python bindings: pip install osmium
try:
    import osmium
except ImportError:
    osmium = None


class WayHandler(osmium.SimpleHandler if osmium else object):
    def __init__(self):
        super().__init__() if osmium else None
        self.nodes = {}
        self.edges = []
        self.node_idx = {}

    def node(self, n):
        self.nodes[n.id] = (n.location.lat, n.location.lon)

    def way(self, w):
        ids = [nd.ref for nd in w.nodes]
        for a, b in zip(ids[:-1], ids[1:]):
            if a in self.nodes and b in self.nodes:
                for nid in (a, b):
                    if nid not in self.node_idx:
                        self.node_idx[nid] = len(self.node_idx)
                self.edges.append([self.node_idx[a], self.node_idx[b]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pbf", required=True)
    ap.add_argument("--out", default="python/hmm/road_graph.json")
    ap.add_argument("--max-nodes", type=int, default=100000)
    args = ap.parse_args()

    if osmium is None:
        raise SystemExit("pip install osmium first (or run on Colab)")

    h = WayHandler()
    h.apply_file(args.pbf, locations=True)

    # keep only nodes touched by edges, cap size
    used = sorted({n for e in h.edges for n in e})[: args.max_nodes]
    remap = {old: new for new, old in enumerate(used)}
    lats = [h.nodes[old][0] for old in used]
    lons = [h.nodes[old][1] for old in used]
    edges = [[remap[a], remap[b]] for a, b in h.edges if a in remap and b in remap]

    out = [{"lat": lats, "lon": lons}, edges]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"[graph] {len(lats)} nodes, {len(edges)} edges -> {args.out} "
          f"({Path(args.out).stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
