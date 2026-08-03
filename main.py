"""Code & Magic compatible standalone Pyxel blue wireframe player.

Plays precomputed video wireframe data without FFmpeg, subprocess, or OpenCV.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyxel


WIDTH = 360
HEIGHT = 640
FPS = 24

# The feature mesh contains 18 columns x 32 rows.
BLOCK = 20
BLOCK_ROWS = HEIGHT // BLOCK
BLOCK_COLS = WIDTH // BLOCK

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "frame_data.npz"
AUDIO_FILE = HERE / "source_audio.ogg"


def make_wire_palette() -> list[int]:
    """Create a near-black to electric-blue colour ramp."""
    colours: list[int] = []

    for index in range(256):
        t = index / 255.0

        red = int(2 + 178 * t**1.80)
        green = int(7 + 229 * t**1.22)
        blue = int(15 + 240 * t**0.76)

        colours.append((red << 16) | (green << 8) | blue)

    colours[0] = 0x01040B
    colours[-1] = 0xDDF8FF

    return colours


WIRE_PALETTE = make_wire_palette()


class App:
    DETAIL_NAMES = ("STRUCTURE", "ARMOR", "MICRO MESH")

    def __init__(self) -> None:
        if not DATA_FILE.is_file():
            raise FileNotFoundError(
                f"Frame data file missing: {DATA_FILE}"
            )

        self.load_frame_data()

        self.frame_number = 0
        self.detail = 1
        self.show_mesh = True
        self.long_echo = False
        self.show_info = False
        self.muted = False
        self.audio_started = False

        self.history = np.zeros(
            (HEIGHT, WIDTH),
            dtype=np.uint8,
        )

        pyxel.init(
            WIDTH,
            HEIGHT,
            title="Knight Arena // Re:0801",
            fps=FPS,
            quit_key=pyxel.KEY_NONE,
            display_scale=1,
        )

        pyxel.colors[:] = WIRE_PALETTE

        self.screen = np.frombuffer(
            pyxel.screen.data_ptr(),
            dtype=np.uint8,
            count=WIDTH * HEIGHT,
        ).reshape((HEIGHT, WIDTH))

        if AUDIO_FILE.is_file():
            pyxel.sounds[0].pcm(str(AUDIO_FILE))

        pyxel.run(self.update, self.draw)

    def load_frame_data(self) -> None:
        """Load and unpack the precomputed frame dataset."""
        with np.load(DATA_FILE, allow_pickle=False) as data:
            if "packed_640x360" in data:
                packed = data["packed_640x360"]

                # packed shape:
                # (frames, detail levels, 640, 180)
                #
                # Each byte stores two 4-bit pixels.
                high = (packed >> 4) * 17
                low = (packed & 0x0F) * 17

                frame_count = len(packed)

                # Directly reconstruct the 360 x 640 version.
                # Do not enlarge it to 720 x 1280.
                lines = np.empty(
                    (frame_count, 3, HEIGHT, WIDTH),
                    dtype=np.uint8,
                )

                lines[..., 0::2] = high
                lines[..., 1::2] = low

                self.lines = lines
            elif "lines" in data:
                lines = data["lines"]

                if lines.shape[-2:] == (1280, 720):
                    # Reduce the original 720 x 1280 data by half.
                    lines = lines[..., ::2, ::2]

                if lines.shape[-2:] != (HEIGHT, WIDTH):
                    raise ValueError(
                        "Unexpected frame dimensions: "
                        f"{lines.shape[-2:]}. "
                        f"Expected {(HEIGHT, WIDTH)}."
                    )

                self.lines = lines.astype(
                    np.uint8,
                    copy=False,
                )
            else:
                raise KeyError(
                    "frame_data.npz does not contain "
                    "'packed_640x360' or 'lines'."
                )

            self.mesh_x = data["mesh_x"].copy()
            self.mesh_y = data["mesh_y"].copy()
            self.mesh_act = data["mesh_act"].copy()
            self.mesh_active = data["mesh_active"].copy()

        self.total_frames = min(
            len(self.lines),
            len(self.mesh_x),
            len(self.mesh_y),
            len(self.mesh_act),
            len(self.mesh_active),
        )

        if self.total_frames <= 0:
            raise ValueError("The frame dataset contains no frames.")

    def start_audio(self) -> None:
        if AUDIO_FILE.is_file() and not self.muted:
            pyxel.play(0, 0, loop=True)

        self.audio_started = True

    def restart(self) -> None:
        pyxel.stop(0)

        self.frame_number = 0
        self.audio_started = False
        self.history.fill(0)

    def update_controls(self) -> bool:
        if (
            pyxel.btnp(pyxel.KEY_ESCAPE)
            or pyxel.btnp(pyxel.KEY_Q)
        ):
            pyxel.quit()
            return False

        if pyxel.btnp(pyxel.KEY_R):
            self.restart()

        if pyxel.btnp(pyxel.KEY_H):
            self.show_info = not self.show_info

        if pyxel.btnp(pyxel.KEY_G):
            self.show_mesh = not self.show_mesh

        if pyxel.btnp(pyxel.KEY_T):
            self.long_echo = not self.long_echo

        if (
            pyxel.btnp(pyxel.KEY_UP)
            or pyxel.btnp(pyxel.KEY_3)
        ):
            self.detail = min(2, self.detail + 1)

        if (
            pyxel.btnp(pyxel.KEY_DOWN)
            or pyxel.btnp(pyxel.KEY_1)
        ):
            self.detail = max(0, self.detail - 1)

        if pyxel.btnp(pyxel.KEY_2):
            self.detail = 1

        if pyxel.btnp(pyxel.KEY_M):
            self.muted = not self.muted

            if self.muted:
                pyxel.stop(0)
            else:
                self.restart()

        # Browser and touch interaction.
        if pyxel.btnp(pyxel.MOUSE_BUTTON_LEFT):
            mouse_y = pyxel.mouse_y

            if mouse_y > HEIGHT - 100:
                self.show_info = not self.show_info
            elif mouse_y < 100:
                self.show_mesh = not self.show_mesh
            else:
                self.detail = (self.detail + 1) % 3

        return True

    def update(self) -> None:
        if not self.update_controls():
            return

        current_lines = self.lines[
            self.frame_number,
            self.detail,
        ]

        decay = 186 if self.long_echo else 58

        faded = (
            self.history.astype(np.uint16)
            * decay
            // 256
        ).astype(np.uint8)

        self.history[:] = np.maximum(
            current_lines,
            faded,
        )

        self.frame_number = (
            self.frame_number + 1
        ) % self.total_frames

        if not self.audio_started:
            self.start_audio()

    def draw_feature_mesh(self) -> None:
        # The mesh coordinates were generated for a
        # 720 x 1280 canvas, so divide them by two.
        x_points = (
            self.mesh_x[
                self.frame_number,
                self.detail,
            ].astype(np.float32)
            * 0.5
        ).astype(np.int16)

        y_points = (
            self.mesh_y[
                self.frame_number,
                self.detail,
            ].astype(np.float32)
            * 0.5
        ).astype(np.int16)

        activity = self.mesh_act[
            self.frame_number,
            self.detail,
        ]

        active = self.mesh_active[
            self.frame_number,
            self.detail,
        ]

        rows = min(BLOCK_ROWS, active.shape[0])
        columns = min(BLOCK_COLS, active.shape[1])

        for row in range(rows):
            for column in range(columns):
                if not active[row, column]:
                    continue

                x0 = int(x_points[row, column])
                y0 = int(y_points[row, column])
                local_activity = float(
                    activity[row, column]
                )

                colour = int(
                    np.clip(
                        22 + local_activity * 0.42,
                        26,
                        104,
                    )
                )

                if (
                    column + 1 < columns
                    and active[row, column + 1]
                ):
                    pyxel.line(
                        x0,
                        y0,
                        int(x_points[row, column + 1]),
                        int(y_points[row, column + 1]),
                        colour,
                    )

                if (
                    row + 1 < rows
                    and active[row + 1, column]
                ):
                    pyxel.line(
                        x0,
                        y0,
                        int(x_points[row + 1, column]),
                        int(y_points[row + 1, column]),
                        colour,
                    )

                if (
                    row + 1 < rows
                    and column + 1 < columns
                    and active[row + 1, column + 1]
                ):
                    alternate = (
                        row
                        + column
                        + self.frame_number // 12
                    ) & 1

                    if alternate:
                        pyxel.line(
                            x0,
                            y0,
                            int(
                                x_points[
                                    row + 1,
                                    column + 1,
                                ]
                            ),
                            int(
                                y_points[
                                    row + 1,
                                    column + 1,
                                ]
                            ),
                            colour,
                        )
                    else:
                        pyxel.line(
                            int(
                                x_points[
                                    row,
                                    column + 1,
                                ]
                            ),
                            int(
                                y_points[
                                    row,
                                    column + 1,
                                ]
                            ),
                            int(
                                x_points[
                                    row + 1,
                                    column,
                                ]
                            ),
                            int(
                                y_points[
                                    row + 1,
                                    column,
                                ]
                            ),
                            colour,
                        )

    def draw(self) -> None:
        self.screen[:] = self.history

        if self.show_mesh:
            self.draw_feature_mesh()

        if self.show_info:
            panel_width = min(WIDTH - 16, 340)

            pyxel.rect(
                8,
                8,
                panel_width,
                32,
                0,
            )

            mesh = (
                "MESH"
                if self.show_mesh
                else "NO MESH"
            )

            echo = (
                "LONG ECHO"
                if self.long_echo
                else "STABLE"
            )

            sound = (
                "MUTED"
                if self.muted
                else "SOUND"
            )

            pyxel.text(
                14,
                14,
                (
                    "RECONSTRUCTED / "
                    f"{self.DETAIL_NAMES[self.detail]} "
                    f"/ {mesh}"
                ),
                245,
            )

            pyxel.text(
                14,
                24,
                (
                    f"{echo} / {sound} / "
                    f"FRAME {self.frame_number:03d}"
                ),
                176,
            )


if __name__ == "__main__":
    App()