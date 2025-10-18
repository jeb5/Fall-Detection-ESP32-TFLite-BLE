#!/usr/bin/env python3
import sys
import re
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

SAMPLE = """25450 -> 25514 : 0
25514 -> 25578 : 0
25578 -> 25642 : 0
8811 -> 8875 : 1
8842 -> 8906 : 1
8836 -> 8900 : 1
8821 -> 8885 : 1
8826 -> 8890 : 1
9057 -> 9121 : 0
9083 -> 9147 : 1
9058 -> 9122 : 1
9072 -> 9136 : 1
9058 -> 9122 : 1
9355 -> 9419 : 1
9361 -> 9425 : 1
9352 -> 9416 : 1
9368 -> 9432 : 1
9355 -> 9419 : 1
10310 -> 10374 : 1
10304 -> 10368 : 0
10306 -> 10370 : 1
10312 -> 10376 : 1
10287 -> 10351 : 1
10448 -> 10512 : 1
10469 -> 10533 : 1
"""

data = sys.stdin.read()
if not data.strip():
    data = SAMPLE

lines = [ln.strip() for ln in data.splitlines() if ln.strip()]
pattern = re.compile(r'^\s*(\d+)\s*->\s*(\d+)\s*:\s*(\d+)\s*$')

ranges = []
for i, line in enumerate(lines):
    m = pattern.match(line)
    if not m:
        print(f"warning: couldn't parse line {i+1}: {line}", file=sys.stderr)
        continue
    a = int(m.group(1)); b = int(m.group(2)); label = int(m.group(3))
    if b < a:
        a, b = b, a
    ranges.append((a, b, label))

if not ranges:
    raise SystemExit("No valid ranges parsed. Exiting.")

print("Parsed ranges:")
for a, b, label in ranges:
    print(f"{a} -> {b} : {label}")

xs = [x for r in ranges for x in (r[0], r[1])]
xmin, xmax = min(xs), max(xs)
pad = max(1, int((xmax - xmin) * 0.02))

fig, ax = plt.subplots(figsize=(12, 2.5))
y0 = 0.0; height = 1.0

color_map = {0: "blue", 1: "red"}
alpha = 0.35

for a, b, label in ranges:
    rect = Rectangle((a, y0), b - a, height, facecolor=color_map.get(label, "grey"),
                     edgecolor=None, alpha=alpha)
    ax.add_patch(rect)

ax.add_line(Line2D([xmin - pad, xmax + pad], [y0 + height/2, y0 + height/2], linewidth=0.5))
ax.set_xlim(xmin - pad, xmax + pad)
ax.set_ylim(y0 - 0.1, y0 + height + 0.1)
ax.set_yticks([])
ax.set_xlabel("Position (number line)")
ax.set_title("Ranges plotted (overlaps darken) — 0=blue, 1=red")

legend_handles = [Rectangle((0,0),1,1,facecolor=color_map[k], alpha=alpha) for k in sorted(color_map.keys())]
legend_labels = [f"{k}" for k in sorted(color_map.keys())]
ax.legend(legend_handles, legend_labels, title="Label", loc="upper right", bbox_to_anchor=(1, 1.25))

plt.tight_layout()
plt.show()
