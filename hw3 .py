import tkinter as tk
import random
import math
import time

try:
    from PIL import Image, ImageDraw, ImageTk, ImageFilter, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Install Pillow for best visuals: pip install Pillow")


# ─── Color Palette ───
POWDER_COLORS = [
    (200, 40, 120),   # Magenta
    (140, 30, 160),   # Purple
    (60, 160, 220),   # Sky blue
    (220, 60, 130),   # Hot pink
    (100, 60, 180),   # Violet
    (80, 190, 230),   # Cyan
    (180, 50, 90),    # Berry
    (160, 80, 200),   # Lavender
]

BG_COLOR = "#0a0a10"
BG_RGB = (10, 10, 16)


def rand(a, b):
    return a + random.random() * (b - a)


def pick(arr):
    return random.choice(arr)


# ─── Splash Screen ───
class SplashScreen:
    def __init__(self, root, on_complete):
        self.root = root
        self.on_complete = on_complete
        self.W = 900
        self.H = 600

        # Center the splash window
        sx = (root.winfo_screenwidth() - self.W) // 2
        sy = (root.winfo_screenheight() - self.H) // 2
        root.geometry(f"{self.W}x{self.H}+{sx}+{sy}")
        root.overrideredirect(True)
        root.configure(bg=BG_COLOR)

        self.canvas = tk.Canvas(root, width=self.W, height=self.H,
                                bg=BG_COLOR, highlightthickness=0)
        self.canvas.pack()

        # Animation state
        self.clouds = []
        self.dust = []
        self.t = 0
        self.phase = 0       # 0=burst, 1=reveal, 2=idle, 3=fadeout
        self.wave_n = 0
        self.max_waves = 7
        self.next_wave = 15
        self.title_alpha = 0.0
        self.sub_alpha = 0.0
        self.load_alpha = 0.0
        self.fade_alpha = 0.0
        self.shimmer = 0
        self.dots = 0

        if HAS_PIL:
            # Create buffer image for smooth blending
            self.buffer = Image.new("RGBA", (self.W, self.H), (*BG_RGB, 255))
            self.overlay = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
            self.photo = None

        self.start_time = time.time()
        self.animate()

    def create_cloud(self, cx, cy, forced_angle=None):
        c = pick(POWDER_COLORS)
        angle = forced_angle + rand(-0.4, 0.4) if forced_angle is not None else random.random() * math.pi * 2
        speed = rand(0.4, 2.5)
        size = rand(40, 120)
        return {
            "x": cx + rand(-10, 10), "y": cy + rand(-10, 10),
            "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
            "size": size, "target": size * rand(2, 4), "grow": rand(0.5, 2),
            "r": c[0], "g": c[1], "b": c[2],
            "alpha": 0, "max_alpha": rand(60, 160),
            "fade_in": rand(2, 6), "fade_out": rand(0.3, 1.2),
            "life": 0, "max_life": rand(120, 300), "drag": rand(0.985, 0.995),
        }

    def create_dust(self, cx, cy):
        c = pick(POWDER_COLORS)
        angle = random.random() * math.pi * 2
        speed = rand(1, 5)
        return {
            "x": cx + rand(-20, 20), "y": cy + rand(-20, 20),
            "vx": math.cos(angle) * speed, "vy": math.sin(angle) * speed,
            "size": rand(1.5, 4), "r": c[0], "g": c[1], "b": c[2],
            "alpha": rand(80, 200), "decay": rand(1, 5), "drag": 0.99,
        }

    def animate(self):
        self.t += 1
        cx, cy = self.W // 2, self.H // 2

        if HAS_PIL:
            self._animate_pil(cx, cy)
        else:
            self._animate_canvas(cx, cy)

        # Phase transitions
        if self.phase == 0:
            if self.t >= self.next_wave and self.wave_n < self.max_waves:
                count = random.randint(5, 12)
                base_angle = random.random() * math.pi * 2
                for j in range(count):
                    a = base_angle + (j / count) * math.pi * 2
                    self.clouds.append(self.create_cloud(cx, cy, a))
                for j in range(20):
                    self.dust.append(self.create_dust(cx, cy))
                self.wave_n += 1
                self.next_wave = self.t + int(rand(8, 20))

            if self.wave_n >= self.max_waves and self.t > self.next_wave + 60:
                self.phase = 1

        elif self.phase == 1:
            self.title_alpha = min(1.0, self.title_alpha + 0.015)
            if self.title_alpha > 0.3:
                self.sub_alpha = min(1.0, self.sub_alpha + 0.012)
            if self.sub_alpha > 0.4:
                self.load_alpha = min(1.0, self.load_alpha + 0.02)
            if self.title_alpha >= 1 and self.load_alpha >= 1:
                self.phase = 2
                self.idle_start = self.t

        elif self.phase == 2:
            self.shimmer += 1.5
            self.dots = (self.dots + 0.025) % 4
            # After 3 seconds idle, fade out
            if self.t - self.idle_start > 180:
                self.phase = 3

        elif self.phase == 3:
            self.fade_alpha = min(1.0, self.fade_alpha + 0.025)
            if self.fade_alpha >= 1.0:
                self.canvas.destroy()
                self.on_complete()
                return

        # Update particles
        for c in self.clouds[:]:
            c["life"] += 1
            c["x"] += c["vx"]
            c["y"] += c["vy"]
            c["vx"] *= c["drag"]
            c["vy"] *= c["drag"]
            if c["size"] < c["target"]:
                c["size"] += c["grow"]
            if c["life"] < c["max_life"] * 0.3:
                c["alpha"] = min(c["max_alpha"], c["alpha"] + c["fade_in"])
            else:
                c["alpha"] = max(0, c["alpha"] - c["fade_out"])
            if c["life"] > c["max_life"] or c["alpha"] <= 0:
                self.clouds.remove(c)

        for d in self.dust[:]:
            d["x"] += d["vx"]
            d["y"] += d["vy"]
            d["vx"] *= d["drag"]
            d["vy"] *= d["drag"]
            d["alpha"] -= d["decay"]
            if d["alpha"] <= 0:
                self.dust.remove(d)

        self.root.after(16, self.animate)

    def _animate_pil(self, cx, cy):
        # Darken buffer slightly each frame (trail effect)
        dark = Image.new("RGBA", (self.W, self.H), (*BG_RGB, 8))
        self.buffer = Image.alpha_composite(self.buffer, dark)

        # Draw clouds
        overlay = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        for c in self.clouds:
            if c["alpha"] <= 0:
                continue
            s = int(c["size"])
            a = int(min(255, c["alpha"]))
            x, y = int(c["x"]), int(c["y"])
            # Draw multiple layers for soft glow
            for i in range(3):
                mult = [1.0, 0.6, 0.3][i]
                sz = int(s * [0.5, 0.8, 1.2][i])
                al = int(a * mult)
                if al > 0 and sz > 0:
                    draw.ellipse(
                        [x - sz, y - sz, x + sz, y + sz],
                        fill=(c["r"], c["g"], c["b"], al)
                    )

        # Draw dust
        for d in self.dust:
            if d["alpha"] <= 0:
                continue
            a = int(min(255, d["alpha"]))
            s = int(d["size"])
            x, y = int(d["x"]), int(d["y"])
            draw.ellipse([x - s, y - s, x + s, y + s],
                         fill=(d["r"], d["g"], d["b"], a))

        # Blur the overlay for softness
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=12))
        self.buffer = Image.alpha_composite(self.buffer, overlay)

        # Build final frame
        frame = self.buffer.copy()
        frame_draw = ImageDraw.Draw(frame)

        # Title
        if self.title_alpha > 0:
            ta = int(255 * self.title_alpha)
            # Try to get a nice font
            fs = min(self.W // 10, 72)
            font = self._get_font(fs, light=True)
            text = "Muni Color"

            bbox = frame_draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            tx = (self.W - tw) // 2
            ty = int(self.H * 0.43) - (bbox[3] - bbox[1]) // 2

            # Glow
            glow_layer = Image.new("RGBA", (self.W, self.H), (0, 0, 0, 0))
            glow_draw = ImageDraw.Draw(glow_layer)
            glow_draw.text((tx, ty), text, fill=(255, 255, 255, ta // 3), font=font)
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=20))
            frame = Image.alpha_composite(frame, glow_layer)
            frame_draw = ImageDraw.Draw(frame)

            # Crisp text
            frame_draw.text((tx, ty), text, fill=(255, 255, 255, ta), font=font)

        # Subtitle
        if self.sub_alpha > 0:
            sa = int(255 * self.sub_alpha * 0.4)
            ss = min(self.W // 55, 13)
            sfont = self._get_font(ss, light=True)
            sub = "M U N I C I P A L   B O N D   A N A L Y T I C S"
            bbox = frame_draw.textbbox((0, 0), sub, font=sfont)
            sw = bbox[2] - bbox[0]
            stx = (self.W - sw) // 2
            sty = int(self.H * 0.43) + min(self.W // 10, 72) // 2 + 20
            frame_draw.text((stx, sty), sub, fill=(255, 255, 255, sa), font=sfont)

        # Loading bar
        if self.load_alpha > 0:
            la = int(255 * self.load_alpha)
            ly = int(self.H * 0.76)
            bw = min(self.W // 5, 150)
            bx = (self.W - bw) // 2

            # Track
            frame_draw.rectangle([bx, ly, bx + bw, ly + 2],
                                 fill=(255, 255, 255, int(la * 0.06)))

            # Sweep
            if self.phase >= 2:
                prog = int(self.shimmer * 0.5) % (bw + 60)
                sw_x = bx + prog - 30
                for i in range(40):
                    px = sw_x + i
                    if bx <= px <= bx + bw:
                        intensity = 1 - abs(i - 20) / 20
                        a = int(la * 0.4 * max(0, intensity))
                        if a > 0:
                            frame_draw.rectangle([px, ly, px + 1, ly + 2],
                                                 fill=(255, 255, 255, a))

            # Processing text
            dot_n = int(self.dots) if self.phase >= 2 else 0
            ls = min(self.W // 60, 11)
            lfont = self._get_font(ls, light=True)
            ltxt = "Processing" + "." * dot_n
            bbox = frame_draw.textbbox((0, 0), ltxt, font=lfont)
            ltw = bbox[2] - bbox[0]
            frame_draw.text(((self.W - ltw) // 2, ly + 14), ltxt,
                            fill=(255, 255, 255, int(la * 0.25)), font=lfont)

        # Fade out overlay
        if self.fade_alpha > 0:
            fade = Image.new("RGBA", (self.W, self.H),
                             (*BG_RGB, int(255 * self.fade_alpha)))
            frame = Image.alpha_composite(frame, fade)

        # Render
        self.photo = ImageTk.PhotoImage(frame)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

    def _animate_canvas(self, cx, cy):
        """Fallback canvas-only animation (no Pillow)."""
        self.canvas.delete("all")

        for c in self.clouds:
            if c["alpha"] <= 0:
                continue
            a = c["alpha"] / 255
            s = int(c["size"])
            x, y = int(c["x"]), int(c["y"])
            color = f"#{c['r']:02x}{c['g']:02x}{c['b']:02x}"
            self.canvas.create_oval(x - s, y - s, x + s, y + s,
                                    fill=color, outline="", stipple="gray25")

        for d in self.dust:
            if d["alpha"] <= 0:
                continue
            s = int(d["size"])
            x, y = int(d["x"]), int(d["y"])
            color = f"#{d['r']:02x}{d['g']:02x}{d['b']:02x}"
            self.canvas.create_oval(x - s, y - s, x + s, y + s,
                                    fill=color, outline="")

        if self.title_alpha > 0:
            a = int(255 * self.title_alpha)
            color = f"#{a:02x}{a:02x}{a:02x}"
            self.canvas.create_text(self.W // 2, int(self.H * 0.43),
                                    text="Muni Color", fill=color,
                                    font=("Segoe UI", 52, "normal"))

        if self.sub_alpha > 0:
            a = int(255 * self.sub_alpha * 0.4)
            color = f"#{a:02x}{a:02x}{a:02x}"
            self.canvas.create_text(self.W // 2, int(self.H * 0.55),
                                    text="MUNICIPAL BOND ANALYTICS", fill=color,
                                    font=("Segoe UI", 10))

        if self.load_alpha > 0:
            a = int(255 * self.load_alpha * 0.25)
            color = f"#{a:02x}{a:02x}{a:02x}"
            dot_n = int(self.dots) if self.phase >= 2 else 0
            self.canvas.create_text(self.W // 2, int(self.H * 0.76),
                                    text="Processing" + "." * dot_n, fill=color,
                                    font=("Segoe UI", 9))

        if self.fade_alpha > 0:
            a = int(255 * self.fade_alpha)
            self.canvas.create_rectangle(0, 0, self.W, self.H,
                                         fill=BG_COLOR, stipple="gray75")

    def _get_font(self, size, light=False):
        families = ["Segoe UI Light", "Segoe UI", "SF Pro Display",
                     "Helvetica Neue", "Helvetica", "Arial"]
        for fam in families:
            try:
                return ImageFont.truetype(fam, size)
            except (OSError, IOError):
                continue
        # Try common paths
        import platform
        paths = []
        if platform.system() == "Windows":
            paths = [
                "C:/Windows/Fonts/segoeuil.ttf",   # Segoe UI Light
                "C:/Windows/Fonts/segoeui.ttf",     # Segoe UI
                "C:/Windows/Fonts/arial.ttf",
            ]
        elif platform.system() == "Darwin":
            paths = [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/SFPro.ttf",
            ]
        else:
            paths = [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Light.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            ]
        for p in paths:
            try:
                return ImageFont.truetype(p, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()
