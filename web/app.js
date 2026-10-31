/**
 * HOVERSLAM // UI Interactivity & Control Bindings
 * Aesthetic: Silent, Mathematical, Exact
 */

document.addEventListener("DOMContentLoaded", () => {
  const sim = window.HoverslamSim;
  if (!sim) return;

  // 1. Flight Mode Toggles (Autopilot vs Manual Pilot)
  const btnAuto = document.getElementById("btn-mode-auto");
  const btnManual = document.getElementById("btn-mode-manual");
  const modeLabel = document.getElementById("flight-mode-label");

  if (btnAuto && btnManual) {
    btnAuto.addEventListener("click", () => {
      sim.setMode("auto");
      btnAuto.className = "px-3 py-1 rounded-full bg-white text-black font-medium transition-all";
      btnManual.className = "px-3 py-1 rounded-full text-zinc-400 hover:text-white transition-all";
      if (modeLabel) modeLabel.textContent = "Autopilot (PPO)";
    });

    btnManual.addEventListener("click", () => {
      sim.setMode("manual");
      btnManual.className = "px-3 py-1 rounded-full bg-white text-black font-medium transition-all";
      btnAuto.className = "px-3 py-1 rounded-full text-zinc-400 hover:text-white transition-all";
      if (modeLabel) modeLabel.textContent = "Manual Pilot (Keyboard)";
    });
  }

  // 2. Simulator Control Bar Buttons
  const btnReset = document.getElementById("btn-sim-reset");
  if (btnReset) {
    btnReset.addEventListener("click", () => sim.reset());
  }

  const btnTogglePad = document.getElementById("btn-toggle-pad");
  const labelPadDrift = document.getElementById("label-pad-drift");
  if (btnTogglePad && labelPadDrift) {
    btnTogglePad.addEventListener("click", () => {
      const active = sim.togglePadDrift();
      labelPadDrift.textContent = active ? "ON" : "OFF";
      labelPadDrift.className = active ? "text-white font-medium" : "text-zinc-500 font-medium";
    });
  }

  const btnToggleWind = document.getElementById("btn-toggle-wind");
  const labelWind = document.getElementById("label-wind-gust");
  if (btnToggleWind && labelWind) {
    btnToggleWind.addEventListener("click", () => {
      const active = sim.toggleWindGusts();
      labelWind.textContent = active ? "ON" : "OFF";
      labelWind.className = active ? "text-white font-medium" : "text-zinc-500 font-medium";
    });
  }

  // 3. Policy Evolution Milestone Scrubber
  const evoCards = document.querySelectorAll(".evo-card");
  const evoRate = document.getElementById("evo-rate");
  const evoSpeed = document.getElementById("evo-speed");
  const evoDrift = document.getElementById("evo-drift");
  const btnLoadEvo = document.getElementById("btn-load-evo-sim");

  const evoData = {
    "100k": { rate: "12.4%", speed: "18.4 m/s", drift: "0.0 m/s (Lost Pad)" },
    "500k": { rate: "48.2%", speed: "6.2 m/s", drift: "5.0 m/s (Overshoot)" },
    "1500k": { rate: "84.5%", speed: "2.1 m/s", drift: "12.0 m/s (Tracking)" },
    "3000k": { rate: "96.8%", speed: "1.12 m/s", drift: "15.0 m/s (Lead Lock)" }
  };

  let currentEvoStep = "3000k";

  evoCards.forEach((card) => {
    card.addEventListener("click", () => {
      const step = card.getAttribute("data-step");
      if (!step || !evoData[step]) return;

      currentEvoStep = step;

      evoCards.forEach((c) => {
        c.classList.remove("active-evo");
      });

      card.classList.add("active-evo");

      const d = evoData[step];
      if (evoRate) evoRate.textContent = d.rate;
      if (evoSpeed) evoSpeed.textContent = d.speed;
      if (evoDrift) evoDrift.textContent = d.drift;

      sim.setCheckpoint(step);
    });
  });

  if (btnLoadEvo) {
    btnLoadEvo.addEventListener("click", () => {
      sim.setCheckpoint(currentEvoStep);
      const simSection = document.getElementById("simulator");
      if (simSection) {
        simSection.scrollIntoView({ behavior: "smooth" });
      }
    });
  }

  // 4. KaTeX Mathematical Typesetting
  const renderMath = () => {
    if (typeof window.katex !== "undefined") {
      document.querySelectorAll(".katex-formula").forEach((el) => {
        const tex = el.getAttribute("data-tex");
        if (tex) {
          try {
            window.katex.render(tex, el, { displayMode: true, throwOnError: false });
          } catch (err) {
            console.error("KaTeX rendering error:", err);
          }
        }
      });
    }
  };

  renderMath();
  window.addEventListener("load", renderMath);

  // 5. Battery & CPU Optimization: Pause hero video when scrolled past
  const heroVideo = document.querySelector(".hero-video-container video");
  if (heroVideo && "IntersectionObserver" in window) {
    const videoObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          heroVideo.play().catch(() => {});
        } else {
          heroVideo.pause();
        }
      });
    }, { threshold: 0.05 });
    videoObserver.observe(heroVideo);

    document.addEventListener("visibilitychange", () => {
      if (document.hidden) heroVideo.pause();
      else heroVideo.play().catch(() => {});
    });
  }
});
