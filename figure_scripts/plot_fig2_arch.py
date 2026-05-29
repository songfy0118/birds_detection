import os
import re
import sys
from graphviz import Digraph


# ----------------- 核心修复代码开始 -----------------
def fix_windows_graphviz_dll():
    """
    针对 Anaconda Windows 环境强制添加 Library/bin 到 PATH。
    解决 'gvplugin_pango.dll' 依赖找不到的问题。
    """
    if os.name != 'nt':
        return

    # 根据你报错信息里的路径，推导出 Anaconda 的 Library\bin 目录
    # 你的路径是: C:\Users\93785\anaconda3\envs\Test2\Library\bin\gvplugin_pango.dll
    # 所以我们要加的是: C:\Users\93785\anaconda3\envs\Test2\Library\bin

    # 尝试自动获取当前环境路径
    current_env = sys.prefix
    library_bin = os.path.join(current_env, "Library", "bin")

    # 如果自动获取不对，就用你报错信息里的绝对路径 (你可以根据需要取消下面这行的注释)
    # library_bin = r"C:\Users\93785\anaconda3\envs\Test2\Library\bin"

    if os.path.exists(library_bin):
        os.environ["PATH"] = library_bin + os.pathsep + os.environ["PATH"]
        print(f"已将 Graphviz DLL 路径加入 PATH: {library_bin}")

        # Python 3.8+ 需要显式添加 DLL 目录
        if hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(library_bin)
            except Exception:
                pass
    else:
        print(f"警告: 找不到路径 {library_bin}，Graphviz 可能仍会报错")


# ----------------- 核心修复代码结束 -----------------

# ----------------- Paper-aligned settings -----------------
MODEL_NAME = "Mini-BirdFormer"
ENC_LAYERS = 2
EMBED_DIM = 96
OUTPUT_DIST = "Student-t Mixture"
EVAL_HORIZONS = "30 / 60 frames"


# -----------------------------------------------------------------------------

def html_box(title: str, lines: list[str], accent: str, fill: str) -> str:
    def to_ascii(s: str) -> str:
        s = s.replace("–", "-").replace("—", "-").replace("×", "x").replace("…", "...")
        s = s.replace("∈", "in").replace("ℝ", "R")
        s = re.sub(r"[^\x00-\x7F]+", "", s)
        return s

    title = to_ascii(title)
    lines = [to_ascii(x) for x in lines]

    rows = "".join(
        f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10">{line}</FONT></TD></TR>'
        for line in lines
    )
    return f"""<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="8">
  <TR>
    <TD BGCOLOR="{accent}" ALIGN="CENTER">
      <FONT COLOR="white" POINT-SIZE="12"><B>{title}</B></FONT>
    </TD>
  </TR>
  <TR>
    <TD BGCOLOR="{fill}">
      <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">
        {rows}
      </TABLE>
    </TD>
  </TR>
</TABLE>
>"""


def main():
    # 1. 先执行修复
    fix_windows_graphviz_dll()

    # 2. 绘图逻辑
    g = Digraph("MiniBirdFormerArch", format="svg", engine="dot")
    g.attr(
        rankdir="LR",
        splines="spline",
        nodesep="0.35",
        ranksep="0.55",
        fontname="Helvetica",
        bgcolor="white"
    )

    g.attr("edge", color="#222222", penwidth="1.6", arrowsize="0.9")
    g.attr("node", fontname="Helvetica")

    with g.subgraph(name="cluster_core") as c:
        c.attr(
            label="Trajectory Forecasting Core (Sec. 3.1-3.3)",
            color="#D1D5DB",
            style="rounded",
            penwidth="1.4",
            fontname="Helvetica",
            fontsize="12"
        )

        input_label = html_box(
            "Input",
            [
                "Past trajectories (tracklets)",
                "X = {p_{t-T+1}, ..., p_t},  p_t in R^2"
            ],
            accent="#111827", fill="#F9FAFB"
        )
        c.node("inp", label=input_label, shape="plain")

        enc_label = html_box(
            f"{MODEL_NAME} Encoder",
            [
                "Lightweight Transformer",
                f"{ENC_LAYERS} layers, d = {EMBED_DIM}",
                "Outputs context features H"
            ],
            accent="#1D4ED8", fill="#EEF2FF"
        )
        c.node("enc", label=enc_label, shape="plain")

        head_label = html_box(
            f"{OUTPUT_DIST} Head (MDN)",
            [
                "Multimodal + heavy-tailed output",
                "Predict mixture parameters",
                "{pi_k, mu_k, Sigma_k, nu_k} for k=1...K"
            ],
            accent="#B91C1C", fill="#FFF1F2"
        )
        c.node("head", label=head_label, shape="plain")

        out_label = html_box(
            "Probabilistic Forecast",
            [
                "p(y|X) = sum_k pi_k * T_{nu_k}(y; mu_k, Sigma_k)",
                "Output: mean trajectory + uncertainty",
                f"Eval horizons: {EVAL_HORIZONS}"
            ],
            accent="#6D28D9", fill="#F5F3FF"
        )
        c.node("out", label=out_label, shape="plain")

        c.edge("inp", "enc")
        c.edge("enc", "head")
        c.edge("head", "out")

    abbrev_label = html_box(
        "Abbrev.",
        [
            "MDN: Mixture Density Network",
            "T_nu: Student-t distribution",
            "nu: degrees of freedom (tail heaviness)"
        ],
        accent="#374151", fill="#FFFFFF"
    )
    g.node("abbr", label=abbrev_label, shape="plain")

    g.attr("edge", style="dashed", color="#6B7280", penwidth="1.2")
    g.edge("out", "abbr", constraint="false")

    out_name = "fig2_minibirdformer_arch_final"
    try:
        g.render(out_name, cleanup=True)
        print(f"Saved: {out_name}.svg")
    except Exception as e:
        print(f"Error rendering: {e}")
        # 如果还是失败，打印 DOT 源码让用户去在线生成
        print("\n\n渲染失败，请复制下面的代码到 http://magjac.com/graphviz-visual-editor/ 生成图片:\n")
        print(g.source)


if __name__ == "__main__":
    main()