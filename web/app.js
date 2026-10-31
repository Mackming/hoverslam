/**
 * HOVERSLAM // UI Interactivity & Control Bindings
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
      btnAuto.className = "px-2.5 py-1 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 font-medium transition-all";
      btnManual.className = "px-2.5 py-1 rounded text-slate-400 hover:text-white border border-transparent transition-all";
      if (modeLabel) modeLabel.textContent = "AUTOPILOT (PPO)";
    });

    btnManual.addEventListener("click", () => {
      sim.setMode("manual");
      btnManual.className = "px-2.5 py-1 rounded bg-sky-500/20 text-sky-300 border border-sky-500/40 font-medium transition-all";
      btnAuto.className = "px-2.5 py-1 rounded text-slate-400 hover:text-white border border-transparent transition-all";
      if (modeLabel) modeLabel.textContent = "MANUAL PILOT (KEYBOARD)";
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
      labelPadDrift.className = active ? "text-emerald-400 font-bold" : "text-slate-400 font-bold";
    });
  }

  const btnToggleWind = document.getElementById("btn-toggle-wind");
  const labelWind = document.getElementById("label-wind-gust");
  if (btnToggleWind && labelWind) {
    btnToggleWind.addEventListener("click", () => {
      const active = sim.toggleWindGusts();
      labelWind.textContent = active ? "ON" : "OFF";
      labelWind.className = active ? "text-amber-400 font-bold" : "text-slate-400 font-bold";
    });
  }

  // 3. Policy Evolution Milestone Scrubber
  const evoCards = document.querySelectorAll(".evo-card");
  const evoRate = document.getElementById("evo-rate");
  const evoSpeed = document.getElementById("evo-speed");
  const evoDrift = document.getElementById("evo-drift");
  const btnLoadEvo = document.getElementById("btn-load-evo-sim");

  const evoData = {
    "100k": { rate: "12.4%", speed: "18.4 m/s", drift: "0.0 m/s (Lost Pad)", label: "100k Steps (Tumbling)" },
    "500k": { rate: "48.2%", speed: "6.2 m/s", drift: "5.0 m/s (Overshoot)", label: "500k Steps (Altitude Brake)" },
    "1500k": { rate: "84.5%", speed: "2.1 m/s", drift: "12.0 m/s (Tracking)", label: "1.5M Steps (Velocity Comp)" },
    "3000k": { rate: "96.8%", speed: "1.12 m/s", drift: "15.0 m/s (Lead Lock)", label: "3.0M Steps (Flagship Master)" }
  };

  let currentEvoStep = "3000k";

  evoCards.forEach((card) => {
    card.addEventListener("click", () => {
      const step = card.getAttribute("data-step");
      if (!step || !evoData[step]) return;

      currentEvoStep = step;

      evoCards.forEach((c) => {
        c.classList.remove("active-evo", "border-amber-500/40", "bg-amber-500/[0.06]", "shadow-[0_0_20px_rgba(245,158,11,0.15)]");
        c.classList.add("border-white/10", "bg-white/[0.02]");
      });

      card.classList.add("active-evo");
      card.classList.remove("border-white/10", "bg-white/[0.02]");

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

  // 4. Developer Terminal Command Tabs
  const cliTabs = document.querySelectorAll(".cli-tab");
  const cliPanels = document.querySelectorAll(".cli-panel");

  cliTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      const targetId = tab.getAttribute("data-target");
      if (!targetId) return;

      cliTabs.forEach((t) => {
        t.className = "cli-tab px-3 py-1.5 rounded text-slate-400 hover:text-white";
      });
      tab.className = "cli-tab px-3 py-1.5 rounded bg-white/10 text-white font-medium border border-white/20";

      cliPanels.forEach((panel) => {
        if (panel.id === targetId) {
          panel.classList.remove("hidden");
        } else {
          panel.classList.add("hidden");
        }
      });
    });
  });

});
