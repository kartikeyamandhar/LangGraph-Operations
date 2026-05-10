from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()  # must be before importing graph/agents
from tracing import init_langsmith_tracing
init_langsmith_tracing()  # must be before importing graph/agents
from graph import build_graph


if __name__ == "__main__":
 
    app = build_graph()

    state = {
        "pdf_path": "data/SeeWeeS Specialty distribution.pdf",
        "csv_path": "data-for-enhancement/Incoming_shipments_14d_multi_corridor.csv",
    }

    final = app.invoke(state)

    report_html = final.get("report_html", "")
    out_path = "report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"\nReport saved to {out_path}")
