import math
import re
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch


# =========================
# Configuration
# =========================

INPUT_FILE = "tree_log.txt"      # Paste your console output into this file
OUTPUT_IMAGE = "tree3_qhd.png"

QHD_WIDTH = 2560
QHD_HEIGHT = 1440
DPI = 100

# label colors
LABEL_COLORS = {
    1: "#4C78A8",   # blue
    2: "#E45756",   # red
    3: "#54A24B",   # green
}
DEFAULT_COLOR = "#999999"


# =========================
# Tree data structure
# =========================

@dataclass
class Node:
    label: int
    children: List["Node"] = field(default_factory=list)
    uid: int = -1


# =========================
# Parsing
# =========================

accepted_re = re.compile(r"^accepted\s+(\d+)\s*$")
tree_line_re = re.compile(r"^(\s*)(\d+)\s*$")


def parse_tree_block(lines: List[str]) -> Optional[Node]:
    """
    Parse a block like:
    2
      2
        3
      3
    where indentation is 2 spaces per level.
    """
    if not lines:
        return None

    stack: List[Node] = []
    root: Optional[Node] = None
    uid_counter = 0

    for line in lines:
        m = tree_line_re.match(line)
        if not m:
            continue

        indent = len(m.group(1))
        label = int(m.group(2))
        depth = indent // 2

        node = Node(label=label, uid=uid_counter)
        uid_counter += 1

        if depth == 0:
            root = node
            stack = [node]
        else:
            while len(stack) > depth:
                stack.pop()

            if not stack:
                raise ValueError(f"Invalid indentation structure near line: {line!r}")

            parent = stack[-1]
            parent.children.append(node)
            stack.append(node)

    return root


def parse_accepted_trees(text: str) -> List[Tuple[int, Node]]:
    """
    Extract trees that follow lines like:
      accepted 1
      ...
      size = 6
      2
        3
        2
    """
    lines = text.splitlines()
    trees: List[Tuple[int, Node]] = []

    current_accept_num: Optional[int] = None
    current_tree_lines: List[str] = []
    collecting_tree = False

    def flush_current():
        nonlocal current_accept_num, current_tree_lines, collecting_tree
        if current_accept_num is not None and current_tree_lines:
            root = parse_tree_block(current_tree_lines)
            if root is not None:
                trees.append((current_accept_num, root))
        current_accept_num = None
        current_tree_lines = []
        collecting_tree = False

    for line in lines:
        line_stripped = line.rstrip("\n")

        m_acc = accepted_re.match(line_stripped)
        if m_acc:
            flush_current()
            current_accept_num = int(m_acc.group(1))
            continue

        if current_accept_num is None:
            continue

        m_tree = tree_line_re.match(line_stripped)
        if m_tree:
            current_tree_lines.append(line_stripped)
            collecting_tree = True
        else:
            if collecting_tree:
                flush_current()

    flush_current()
    return trees


# =========================
# Layout
# =========================

def compute_positions(root: Node) -> Dict[int, Tuple[float, float]]:
    """
    Compute a tidy top-down tree layout.
    Leaves get consecutive x-positions; internal nodes are centered.
    """
    positions: Dict[int, Tuple[float, float]] = {}
    next_x = [0]

    def walk(node: Node, depth: int):
        if not node.children:
            x = next_x[0]
            next_x[0] += 1
        else:
            child_xs = []
            for child in node.children:
                walk(child, depth + 1)
                child_xs.append(positions[child.uid][0])
            x = sum(child_xs) / len(child_xs)

        y = -depth
        positions[node.uid] = (x, y)

    walk(root, 0)
    return positions


def tree_depth(root: Node) -> int:
    if not root.children:
        return 1
    return 1 + max(tree_depth(ch) for ch in root.children)


# =========================
# Drawing
# =========================

def draw_tree(ax, root: Node, title: str):
    positions = compute_positions(root)

    def draw_edges(node: Node):
        x1, y1 = positions[node.uid]
        for child in node.children:
            x2, y2 = positions[child.uid]
            ax.plot([x1, x2], [y1, y2], linewidth=1.4)
            draw_edges(child)

    def draw_nodes(node: Node):
        x, y = positions[node.uid]
        color = LABEL_COLORS.get(node.label, DEFAULT_COLOR)
        circle = Circle((x, y), radius=0.18, facecolor=color, edgecolor="black", linewidth=1.0)
        ax.add_patch(circle)
        ax.text(x, y, str(node.label), ha="center", va="center", fontsize=8, color="white", weight="bold")
        for child in node.children:
            draw_nodes(child)

    draw_edges(root)
    draw_nodes(root)

    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]

    margin_x = 0.6
    margin_y = 0.5
    ax.set_xlim(min(xs) - margin_x, max(xs) + margin_x)
    ax.set_ylim(min(ys) - margin_y, max(ys) + margin_y)

    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    ax.set_title(title, fontsize=10, pad=4)

def count_nodes(root: Node) -> int:
    return 1 + sum(count_nodes(ch) for ch in root.children)

def choose_trees_per_page(trees):
    max_nodes = max(count_nodes(root) for _, root in trees)

    if max_nodes > 60:
        return 1
    elif max_nodes > 30:
        return 4
    elif max_nodes > 15:
        return 6
    else:
        return 12

def render_pages(trees):
    trees_per_page = choose_trees_per_page(trees)

    n = len(trees)
    page_count = math.ceil(n / trees_per_page)

    figsize = (QHD_WIDTH / DPI, QHD_HEIGHT / DPI)

    for page_idx in range(page_count):
        start = page_idx * trees_per_page
        end = min(start + trees_per_page, n)
        chunk = trees[start:end]

        k = len(chunk)

        # simple layouts depending on number per page
        if k == 1:
            rows, cols = 1, 1
        elif k <= 2:
            rows, cols = 1, 2
        elif k <= 4:
            rows, cols = 2, 2
        elif k <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 4

        fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=DPI)
        fig.suptitle(
            f"TREE(3) accepted sequence – page {page_idx + 1}/{page_count}",
            fontsize=18
        )

        if rows == 1 and cols == 1:
            axes = [[axes]]
        elif rows == 1:
            axes = [axes]
        elif cols == 1:
            axes = [[ax] for ax in axes]

        flat_axes = [ax for row in axes for ax in row]

        for ax, (idx, root) in zip(flat_axes, chunk):
            n_nodes = count_nodes(root)
            draw_tree(ax, root, title=f"#{idx}  ({n_nodes} nodes)")

        for ax in flat_axes[len(chunk):]:
            ax.axis("off")

        legend_handles = [
            Patch(facecolor=LABEL_COLORS.get(1, DEFAULT_COLOR), edgecolor="black", label="label 1"),
            Patch(facecolor=LABEL_COLORS.get(2, DEFAULT_COLOR), edgecolor="black", label="label 2"),
            Patch(facecolor=LABEL_COLORS.get(3, DEFAULT_COLOR), edgecolor="black", label="label 3"),
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=11)

        plt.tight_layout(rect=[0.01, 0.05, 0.99, 0.95])

        filename = f"tree_page_{page_idx + 1:02d}.png"
        fig.savefig(filename, dpi=DPI)
        print(f"Saved {filename}")

    plt.show()

# =========================
# Main
# =========================

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    trees = parse_accepted_trees(text)

    if not trees:
        print("No accepted trees found.")
        return

    render_pages(trees)

if __name__ == "__main__":
    main()