import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Circle, Rectangle

# ----------------------------
# Config (edit here)
# ----------------------------
OUT_DIR = r"C:\Users\93785\Desktop\bird_detection\day1\generator\2"  # 改成你想保存的位置
OUT_NAME = "mini_birdformer_arch_clean.png"
DPI = 300

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "stix"
plt.rcParams["font.size"] = 12


# ----------------------------
# Helper drawing functions
# ----------------------------
def add_round_box(ax, x, y, w, h, fc="#FFFFFF", ec="#000000", lw=1.8, text="",
                  fontsize=12, weight="normal", align="center", pad=0.02,
                  boxstyle="round,pad=0.02,rounding_size=0.08", z=2):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=boxstyle,
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=z
    )
    ax.add_patch(patch)

    if text:
        ha = "center" if align == "center" else "left"
        tx = x + w / 2 if ha == "center" else x + w * pad
        ty = y + h / 2
        ax.text(tx, ty, text, ha=ha, va="center", fontsize=fontsize, fontweight=weight, zorder=z + 1)

    return patch


def add_text(ax, x, y, s, fontsize=11, weight="normal", ha="center", va="center", z=5):
    ax.text(x, y, s, fontsize=fontsize, fontweight=weight, ha=ha, va=va, zorder=z)


def add_ellipse(ax, cx, cy, w, h, fc="#FFFFFF", ec="#000000", lw=1.8, text="",
                fontsize=12, weight="normal", z=3):
    patch = Ellipse((cx, cy), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(patch)
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, zorder=z + 1)
    return patch


def add_circle(ax, cx, cy, r, fc="#FFFFFF", ec="#000000", lw=1.6, text="",
               fontsize=12, weight="bold", z=4):
    patch = Circle((cx, cy), r, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    ax.add_patch(patch)
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, fontweight=weight, zorder=z + 1)
    return patch


def add_arrow(ax, x1, y1, x2, y2, color="#4C5A67", lw=1.8, head=10, z=6):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=head),
        zorder=z
    )


def add_dashed_container(ax, x, y, w, h, ec="#2E75D7", lw=2.0, dash=(5, 3), z=1):
    rect = Rectangle((x, y), w, h, fill=False, edgecolor=ec, linewidth=lw,
                     linestyle=(0, dash), zorder=z)
    ax.add_patch(rect)
    return rect


def add_label_box(ax, x, y, w, h, text, fc="#FFFFFF", ec="#808080", lw=1.0, fontsize=10):
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=10
    )
    ax.add_patch(patch)
    ax.text(x + w * 0.05, y + h / 2, text, ha="left", va="center", fontsize=fontsize, zorder=11)
    return patch


# ----------------------------
# Main figure
# ----------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, OUT_NAME)

    fig = plt.figure(figsize=(14, 5))
    ax = plt.gca()
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    # ----------------------------
    # Palette (coherent & paper-friendly)
    # ----------------------------
    purple_ec, purple_fc = "#6F42C1", "#F2ECFB"
    green_ec, green_fc = "#1B7F6A", "#DDF3EC"
    blue_ec, blue_fc = "#2F6FDB", "#DCEBFF"
    red_ec, red_fc = "#C64545", "#F8E1E1"

    gray = "#5B6773"
    light_gray = "#A7B0BA"

    ell_fc1, ell_ec1 = "#E8E1F4", "#7B6DA8"   # pi
    ell_fc2, ell_ec2 = "#E3E8F6", "#6C77A5"   # mu
    ell_fc3, ell_ec3 = "#D9EEFF", "#4C8EC8"   # Sigma
    ell_fc4, ell_ec4 = "#F8E6C8", "#C28B2A"   # nu

    # ----------------------------
    # 1) Input trajectory block
    # ----------------------------
    add_round_box(
        ax, x=0.6, y=2.0, w=2.2, h=1.0,
        fc=purple_fc, ec=purple_ec, lw=2.4,
        text="Trajectory\n$X=\\{\\Delta p_t\\}$",
        fontsize=13, weight="bold"
    )

    # ----------------------------
    # 2) Linear embed block
    # ----------------------------
    add_round_box(
        ax, x=3.2, y=2.0, w=1.7, h=1.0,
        fc=green_fc, ec=green_ec, lw=2.4,
        text="Linear\n\\,\\,Embed",
        fontsize=13, weight="bold"
    )

    # plus circle near linear top
    add_circle(ax, cx=3.9, cy=3.25, r=0.14, fc="#FFFFFF", ec=gray, lw=1.6, text="+", fontsize=12, weight="bold")

    # ----------------------------
    # 3) PE node
    # ----------------------------
    add_ellipse(ax, cx=3.9, cy=3.65, w=0.7, h=0.45, fc="#FFF2CC", ec="#F1B400", lw=2.0,
                text="PE", fontsize=12, weight="bold")

    add_arrow(ax, 3.9, 3.45, 3.9, 3.38, color=gray, lw=1.7, head=10)
    add_arrow(ax, 4.1, 2.95, 3.97, 3.12, color=gray, lw=1.7, head=10)

    # ----------------------------
    # 4) Encoder container (dashed)
    # ----------------------------
    enc_x, enc_y, enc_w, enc_h = 5.3, 1.55, 3.6, 2.2
    add_dashed_container(ax, enc_x, enc_y, enc_w, enc_h, ec=blue_ec, lw=2.2, dash=(5, 3))

    layer_w, layer_h = 3.0, 0.65

    add_round_box(
        ax, x=5.6, y=3.0, w=layer_w, h=layer_h,
        fc=blue_fc, ec=blue_ec, lw=2.0,
        text="Layer 2\nMHSA + FFN",
        fontsize=12, weight="bold"
    )
    add_round_box(
        ax, x=5.6, y=2.05, w=layer_w, h=layer_h,
        fc=blue_fc, ec=blue_ec, lw=2.0,
        text="Layer 1\nMHSA + FFN",
        fontsize=12, weight="bold"
    )

    add_arrow(ax, 7.1, 2.72, 7.1, 2.98, color=gray, lw=1.6, head=10)

    add_text(ax, x=7.1, y=1.65, s="Mini-BirdFormer Encoder", fontsize=11, weight="bold", ha="center", va="center", z=8)
    ax.plot([5.6, 8.6], [1.75, 1.75], color=blue_ec, lw=1.2, zorder=7)

    # ----------------------------
    # 5) MHSA/FFN callout (upper-right, not blocking main flow)
    # ----------------------------
    call_x, call_y = 9.05, 3.95
    add_label_box(
        ax, call_x, call_y, 2.55, 0.55,
        "MHSA: Multi-Head Self-Attention\nFFN: Feed-Forward Network",
        fc="#FFFFFF", ec=light_gray, lw=1.0, fontsize=9
    )
    add_arrow(ax, 8.55, 3.45, call_x, call_y + 0.20, color=light_gray, lw=1.0, head=8)

    # ----------------------------
    # 6) MDN Head block
    # ----------------------------
    add_round_box(
        ax, x=9.6, y=2.05, w=1.5, h=1.35,
        fc=red_fc, ec=red_ec, lw=2.4,
        text="MDN Head\nMLP",
        fontsize=12, weight="bold"
    )

    # Encoder -> MDN (straighter, avoid callouts)
    add_arrow(ax, 8.9, 2.72, 9.6, 2.72, color=gray, lw=1.8, head=10)

    # ----------------------------
    # 7) Output parameter ellipses
    # ----------------------------
    px = 12.1
    y_pi, y_mu, y_sig, y_nu = 3.7, 3.05, 2.4, 1.75

    add_ellipse(ax, px, y_pi, 0.85, 0.55, fc=ell_fc1, ec=ell_ec1, lw=2.0, text="$\\pi$", fontsize=14, weight="bold")
    add_text(ax, px + 0.9, y_pi, "Weights", fontsize=11, ha="left", va="center")

    add_ellipse(ax, px, y_mu, 0.85, 0.55, fc=ell_fc2, ec=ell_ec2, lw=2.0, text="$\\mu$", fontsize=14, weight="bold")
    add_text(ax, px + 0.9, y_mu, "Means", fontsize=11, ha="left", va="center")

    add_ellipse(ax, px, y_sig, 0.85, 0.55, fc=ell_fc3, ec=ell_ec3, lw=2.0, text="$\\Sigma$", fontsize=14, weight="bold")
    add_text(ax, px + 0.9, y_sig, "Scales", fontsize=11, ha="left", va="center")

    add_ellipse(ax, px, y_nu, 0.85, 0.55, fc=ell_fc4, ec=ell_ec4, lw=2.2, text="$\\nu$", fontsize=14, weight="bold")
    add_text(ax, px + 0.9, y_nu, "DOF", fontsize=11, ha="left", va="center")

    mdn_out_x, mdn_out_y = 11.1, 2.72
    add_arrow(ax, mdn_out_x, mdn_out_y, px - 0.45, y_pi, color=gray, lw=1.6, head=10)
    add_arrow(ax, mdn_out_x, mdn_out_y, px - 0.45, y_mu, color=gray, lw=1.6, head=10)
    add_arrow(ax, mdn_out_x, mdn_out_y, px - 0.45, y_sig, color=gray, lw=1.6, head=10)
    add_arrow(ax, mdn_out_x, mdn_out_y, px - 0.45, y_nu, color=gray, lw=1.6, head=10)

    # ----------------------------
    # 8) PE legend (smaller; near PE)
    # ----------------------------
    add_label_box(
        ax, x=2.35, y=4.30, w=2.05, h=0.42,
        text="PE: Positional Encoding",
        fc="#FFFFFF", ec=light_gray, lw=1.0, fontsize=10
    )
    add_arrow(ax, 2.95, 4.30, 3.62, 3.83, color=light_gray, lw=1.0, head=8)

    # ----------------------------
    # 9) Connect Trajectory -> Linear; plus -> Encoder
    # ----------------------------
    add_arrow(ax, 2.8, 2.5, 3.2, 2.5, color=gray, lw=1.8, head=10)
    add_arrow(ax, 4.05, 2.85, 5.6, 2.62, color=gray, lw=1.8, head=10)

    # ----------------------------
    # 10) MDN/DOF footnote (bottom center)
    # ----------------------------
    add_label_box(
        ax, x=7.8, y=0.55, w=4.2, h=0.48,
        text="MDN: Mixture Density Network    DOF: Degrees of Freedom",
        fc="#FFFFFF", ec=light_gray, lw=1.0, fontsize=9
    )

    plt.tight_layout(pad=0.2)
    plt.savefig(out_path, dpi=DPI, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
