import sys
import subprocess
from pathlib import Path

from PySide6.QtCore import QThread, Signal, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QStackedLayout,
    QFrame,
    QAbstractItemView,
)

from reproduction_commands import COMMANDS


EXPERIMENT_GROUPS = {
    "IoT-23": [
        "iot_sample_local_test",
        "iot_sample_prep",
        "iot_rf_baseline",
        "iot_rf_eval",
        "iot_loso_rf",
        "iot_full_prep",
        "iot_feature_stability_plots",
    ],
    "Cross-dataset": [
        "cross_dataset_eval",
        "cross_dataset_plots",
    ],
    "UNSW-NB15": [
        "unsw_inspect",
        "unsw_xgb_baseline",
        "unsw_rf_baseline",
        "unsw_xgb_l1ao",
        "unsw_rf_l1ao",
        "unsw_l1ao_analysis",
        "unsw_l1ao_plots",
    ],
}

HEADER_COLORS = {
    "IoT-23": "#7ee787",
    "Cross-dataset": "#79c0ff",
    "UNSW-NB15": "#ffa657",
}


SPRITE = [
    "000000000000001111110000000111111000000000000",
    "000000000011000111111100011111100110000000000",
    "000000001111110000000000000000001111100000000",
    "000000011111100000000000000000000111110000000",
    "000000011111100000000000000000000111110000000",
    "000000001110000001111111110000000011100000000",
    "000000000000000111111111111100000000000000000",
    "001111000000001111111111111111000000001111000",
    "011111100001111111001111001111110000111111100",
    "111111110011111111001111001111111001111111110",
    "110000000011111111001111001111111000000000110",
    "110000000011111111001111001111111000000000110",
    "111111100011111111001111001111111100011111110",
    "011111110011111111001111001111111000111111100",
    "001111110011110011111111111100110011111111000",
    "000111100001111000000000000111100001111100000",
    "000011110001111110011111001111110001111000000",
    "000000000000111111000000111111000000000000000",
    "000000000000111111000000111111000000000000000",
    "000000000001000111111111111110001000000000000",
    "000000001111000011111111111100001111000000000",
    "000000011111000000011111110000000111110000000",
    "000000111100110000000000000000110011110000000",
    "000000111001111110000000000111111001110000000",
    "000000111001111111111001111111111001110000000",
    "000000111001111111111001111111111001110000000",
    "000000111111111111110000001111111111110000000",
    "000000001111111000000000000000111111100000000",
    "000000000000000000000000000000000000000000000",
    "000000000000000000000110000000000000000000000",
    "000000000000000000000111100000000000000000000",
    "000000000000000000000111100000000000000000000",
    "000000000000000000000111100000000000000000000",
    "000000000000000000001111000000000000000000000",
    "000000000000000000001110000000000000000000000",
    "000000000000000000001110000000000000000000000",
    "000000000000000000000111000000000000000000000",
    "000000000000000000000111110000000000000000000",
    "000000000000000000000011111000000000000000000",
    "000000000000000000000011111000000000000000000",
    "000000000000000000000001111000000000000000000",
    "000000000000000000000000111100000000000000000",
    "000000000000000000000000111100000000000000000",
    "000000000000000000000001111110000000000000000",
    "000000000000000010001111111110001000000000000",
    "000000000000000010011111111110011000000000000",
    "000000000000000010011111111110011000000000000",
    "000000000000000001001111111100100000000000000",
    "000000000000000000100000000001000000000000000",
    "000000000000000000011111111110000000000000000",
]

def render_sprite() -> str:
    lines = []
    for row in SPRITE:
        line = "".join("██" if c == "1" else "  " for c in row)
        lines.append(line)
    return "\n".join(lines)


def render_sprite() -> str:
    lines = []
    for row in SPRITE:
        line = "".join("██" if c == "1" else "  " for c in row)
        lines.append(line)
    return "\n".join(lines)


class CommandWorker(QThread):
    output = Signal(str)
    finished_ok = Signal(int)

    def __init__(self, cmd, cwd):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd

    def run(self):
        process = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=self.cwd,
        )

        assert process.stdout is not None
        for line in process.stdout:
            self.output.emit(line.rstrip("\n"))

        process.wait()
        self.finished_ok.emit(process.returncode)


class ExperimentGUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bachelor Thesis Reproduction Interface")
        self.resize(1200, 760)

        self.current_key = None
        self.worker = None

        self.stack = QStackedLayout(self)
        self.stack.addWidget(self.build_intro())
        self.stack.addWidget(self.build_main())

    def build_intro(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(16)

        layout.addStretch()

        flower = QLabel(render_sprite())
        flower.setAlignment(Qt.AlignCenter)
        flower_font = QFont("Courier New")
        flower_font.setStyleHint(QFont.Monospace)
        flower_font.setPointSize(8)
        flower.setFont(flower_font)
        flower.setObjectName("flower")

        title = QLabel("Bachelor Thesis Reproduction Interface")
        title.setAlignment(Qt.AlignCenter)
        title.setObjectName("introTitle")

        subtitle = QLabel(
            "Howdy! This interface lets you reproduce the experiments directly!"
        )
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setObjectName("introSubtitle")

        start_btn = QPushButton("Start Experiments")
        start_btn.setFixedWidth(220)
        start_btn.setObjectName("startButton")
        start_btn.clicked.connect(lambda: self.stack.setCurrentIndex(1))

        layout.addWidget(flower)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(start_btn, alignment=Qt.AlignCenter)
        layout.addStretch()

        return widget

    def build_main(self):
        widget = QWidget()
        shell = QVBoxLayout(widget)
        shell.setContentsMargins(24, 20, 24, 20)
        shell.setSpacing(0)

        content = QWidget()
        content.setObjectName("mainShell")
        content.setMaximumWidth(1680)

        outer = QHBoxLayout(content)
        outer.setContentsMargins(20, 20, 20, 20)
        outer.setSpacing(16)

        # Left panel
        left_panel = QFrame()
        left_panel.setObjectName("panel")
        left_panel.setMinimumWidth(380)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(14, 14, 14, 14)
        left_layout.setSpacing(10)

        left_title = QLabel("Experiments")
        left_title.setObjectName("panelTitle")

        self.list_widget = QListWidget()
        self.list_widget.setObjectName("experimentList")
        self.list_widget.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list_widget.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.list_widget.setSpacing(4)

        # grouped flat list with colored headers
        for group_name, keys in EXPERIMENT_GROUPS.items():
            header = QListWidgetItem(group_name)
            header.setFlags(Qt.NoItemFlags)
            header.setData(Qt.UserRole, None)
            header_font = QFont(self.list_widget.font())
            header_font.setBold(True)
            header_font.setPointSize(header_font.pointSize() + 1)
            header.setFont(header_font)
            header.setForeground(QBrush(QColor(HEADER_COLORS[group_name])))
            header.setSizeHint(header.sizeHint() + QSize(0, 10))
            self.list_widget.addItem(header)

            for key in keys:
                item = QListWidgetItem(f"  {COMMANDS[key]['label']}")
                item.setData(Qt.UserRole, key)
                self.list_widget.addItem(item)

        self.list_widget.currentItemChanged.connect(self.show_details)

        left_layout.addWidget(left_title)
        left_layout.addWidget(self.list_widget)

        # Right panel
        right_panel = QFrame()
        right_panel.setObjectName("panel")
        right_panel.setMinimumWidth(700)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        details_title = QLabel("Experiment Details")
        details_title.setObjectName("panelTitle")

        self.run_button = QPushButton("Run Selected")
        self.run_button.setObjectName("runButton")
        self.run_button.clicked.connect(self.run_selected)
        self.run_button.setEnabled(False)

        self.back_button = QPushButton("Back")
        self.back_button.setObjectName("secondaryButton")
        self.back_button.clicked.connect(lambda: self.stack.setCurrentIndex(0))

        top_row.addWidget(details_title)
        top_row.addStretch()
        top_row.addWidget(self.back_button)
        top_row.addWidget(self.run_button)

        self.description = QLabel("Select an experiment from the left.")
        self.description.setObjectName("description")
        self.description.setWordWrap(True)
        self.description.setTextFormat(Qt.RichText)
        self.description.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        log_title = QLabel("Live Output")
        log_title.setObjectName("panelTitle")

        self.log = QTextEdit()
        self.log.setObjectName("logBox")
        self.log.setReadOnly(True)

        right_layout.addLayout(top_row)
        right_layout.addWidget(self.description)
        right_layout.addWidget(log_title)
        right_layout.addWidget(self.log)

        outer.addWidget(left_panel, 2)
        outer.addWidget(right_panel, 5)
        shell.addWidget(content, alignment=Qt.AlignHCenter | Qt.AlignTop)

        return widget

    def show_details(self, current, previous):
        if not current:
            self.current_key = None
            self.run_button.setEnabled(False)
            return

        key = current.data(Qt.UserRole)
        if not key:
            self.current_key = None
            self.run_button.setEnabled(False)
            return

        cmd = COMMANDS[key]
        self.current_key = key
        self.run_button.setEnabled(True)

        command_text = " ".join(cmd["cmd"])
        self.description.setText(
            f"<div style='font-size:16px; font-weight:600; margin-bottom:8px;'>{cmd['label']}</div>"
            f"<div style='color:#b8c0cc; margin-bottom:12px;'>{cmd['description']}</div>"
            f"<div style='color:#7ee787; font-weight:600;'>Command</div>"
            f"<div style='font-family:Consolas, monospace; background:#0d1117; "
            f"border:1px solid #30363d; border-radius:8px; padding:10px; margin-top:6px;'>"
            f"{command_text}</div>"
        )

    def run_selected(self):
        if not self.current_key:
            return

        cmd = COMMANDS[self.current_key]["cmd"]
        cwd = Path(__file__).resolve().parent.parent

        self.run_button.setEnabled(False)
        self.log.append(f"$ {' '.join(cmd)}")
        self.log.append("")

        self.worker = CommandWorker(cmd, cwd)
        self.worker.output.connect(self.append_log_line)
        self.worker.finished_ok.connect(self.command_finished)
        self.worker.start()

    def append_log_line(self, line: str):
        self.log.append(line)

    def command_finished(self, return_code: int):
        self.log.append("")
        self.log.append(f"[exit code: {return_code}]")
        self.log.append("")
        self.run_button.setEnabled(True)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    app.setStyleSheet("""
    QWidget {
        background-color: #0d1117;
        color: #e6edf3;
        font-family: "Segoe UI";
        font-size: 13px;
    }

    QFrame#panel {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 14px;
    }

    QWidget#mainShell {
        background: transparent;
    }

    QLabel#introTitle {
        font-size: 28px;
        font-weight: 700;
        color: #f0f6fc;
        margin-top: 10px;
    }

    QLabel#introSubtitle {
        font-size: 14px;
        color: #9da7b3;
        margin-bottom: 10px;
    }

    QLabel#flower {
        color: #7ee787;
    }

    QLabel#panelTitle {
        font-size: 16px;
        font-weight: 600;
        color: #f0f6fc;
    }

    QLabel#description {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 14px;
    }

    QPushButton {
        border: none;
        border-radius: 10px;
        padding: 10px 16px;
        font-weight: 600;
    }

    QPushButton#startButton {
        background-color: #2ea043;
        color: white;
        min-height: 42px;
    }

    QPushButton#startButton:hover {
        background-color: #3fb950;
    }

    QPushButton#runButton {
        background-color: #1f6feb;
        color: white;
        min-height: 40px;
    }

    QPushButton#runButton:hover {
        background-color: #388bfd;
    }

    QPushButton#runButton:disabled {
        background-color: #30363d;
        color: #8b949e;
    }

    QPushButton#secondaryButton {
        background-color: #21262d;
        color: #e6edf3;
        min-height: 40px;
    }

    QPushButton#secondaryButton:hover {
        background-color: #30363d;
    }

    QListWidget#experimentList {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 12px;
        outline: 0;
        padding: 6px;
        min-height: 520px;
    }

    QListWidget#experimentList::item {
        padding: 8px 10px;
        border-radius: 8px;
        margin: 2px 0px;
    }

    QListWidget#experimentList::item:hover {
        background-color: #161b22;
    }

    QListWidget#experimentList::item:selected {
        background-color: #1f6feb;
        color: white;
    }

    QScrollBar:vertical {
        background: #0d1117;
        width: 12px;
        margin: 8px 4px 8px 0;
        border-radius: 6px;
    }

    QScrollBar::handle:vertical {
        background: #30363d;
        min-height: 28px;
        border-radius: 6px;
    }

    QScrollBar::handle:vertical:hover {
        background: #484f58;
    }

    QScrollBar::add-line:vertical,
    QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {
        background: none;
        height: 0;
    }

    QTextEdit#logBox {
        background-color: #0d1117;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 10px;
        font-family: "Consolas";
        font-size: 12px;
    }
    """)

    window = ExperimentGUI()
    window.showMaximized()
    sys.exit(app.exec())
