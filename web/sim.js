/**
 * HOVERSLAM // Real-time Kinematic Physics & Autopilot Engine
 * Aesthetic: Stark Monochrome + Electric Volt (#d4ff00)
 */

(function () {
  const canvas = document.getElementById("flight-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");

  // World & Physics Constants
  const WORLD_W = 800.0;
  const WORLD_H = 600.0;
  const GRAVITY = -45.0;
  const MAIN_ENGINE_POWER = 126.0;
  const SIDE_ENGINE_POWER = 12.0;
  const ANGULAR_DAMPING = 0.94;
  const LINEAR_DAMPING = 0.999;

  const ROCKET_HEIGHT = 34.0;
  const ROCKET_WIDTH = 14.0;
  const LEG_SPAN = 22.0;
  const FOOT_DROP = 6.0;
  const FOOT_OFFSET = ROCKET_HEIGHT / 2.0 + FOOT_DROP; // 23.0

  const PAD_HALF_WIDTH = 60.0;
  const PAD_HEIGHT = 14.0;
  const PAD_MOVE_SPEED = 15.0;
  const PAD_MOVE_RANGE = 180.0;
  const MAX_IMPACT_SPEED = 63.0;
  const TIP_ANGLE = Math.atan2(LEG_SPAN / 2.0, FOOT_OFFSET); // ~25.6 deg (0.447 rad)

  const FPS = 60;
  const DT = 1.0 / FPS;

  // Simulator State
  const sim = {
    mode: "auto", // "auto" or "manual"
    checkpoint: "3000k",
    padDrift: true,
    windGusts: false,

    // Rocket State
    x: 400.0,
    y: 520.0,
    vx: 0.0,
    vy: 0.0,
    angle: 0.0,
    angularVel: 0.0,
    fuel: 100.0,

    // Pad State
    padX: 400.0,
    padVx: PAD_MOVE_SPEED,
    padDir: 1.0,

    // Controls
    mainThrottle: 0.0,
    sideThrottle: 0.0,

    // Episode & Outcome
    landed: false,
    crashed: false,
    grounded: false,
    outcomeHold: 0,
    screenShake: 0,
    shakeX: 0,
    shakeY: 0,

    // Contacts
    legLeftContact: false,
    legRightContact: false,

    // Visual particles
    particles: [],
    stars: []
  };

  // Generate minimalist starfield
  for (let i = 0; i < 80; i++) {
    sim.stars.push({
      x: Math.random() * WORLD_W,
      y: Math.random() * (WORLD_H - 100),
      r: Math.random() * 1.2 + 0.4,
      alpha: Math.random() * 0.5 + 0.2
    });
  }

  // Keyboard state for Manual Flight
  const keys = { ArrowUp: false, ArrowLeft: false, ArrowRight: false };

  window.addEventListener("keydown", (e) => {
    if (e.key in keys) {
      keys[e.key] = true;
      e.preventDefault();
    } else if (e.key === "r" || e.key === "R") {
      resetRocket();
    }
  });

  window.addEventListener("keyup", (e) => {
    if (e.key in keys) {
      keys[e.key] = false;
      e.preventDefault();
    }
  });

  function resetRocket() {
    sim.x = 220.0 + Math.random() * 360.0;
    sim.y = 490.0 + Math.random() * 60.0;
    sim.vx = (Math.random() - 0.5) * 4.0;
    sim.vy = -Math.random() * 5.0;
    sim.angle = (Math.random() - 0.5) * 0.12;
    sim.angularVel = 0.0;
    sim.fuel = 100.0;

    sim.mainThrottle = 0.0;
    sim.sideThrottle = 0.0;
    sim.landed = false;
    sim.crashed = false;
    sim.grounded = false;
    sim.outcomeHold = 0;
    sim.screenShake = 0;
    sim.legLeftContact = false;
    sim.legRightContact = false;

    sim.padX = WORLD_W / 2.0 + (Math.random() - 0.5) * 100.0;
    sim.padDir = Math.random() < 0.5 ? 1.0 : -1.0;
    sim.padVx = sim.padDir * PAD_MOVE_SPEED;

    hideVerdict();
  }

  // Autonomous Guidance Controller (Simulates PPO Policy)
  function computeAutopilotControls() {
    if (sim.landed || sim.crashed) {
      return { main: 0.0, side: 0.0 };
    }

    const altitude = sim.y - PAD_HEIGHT;
    const targetX = sim.padX;
    const dx = sim.x - targetX;

    if (sim.checkpoint === "100k") {
      const main = Math.random() > 0.35 ? 0.95 : 0.05;
      const side = (Math.random() - 0.5) * 1.6;
      return { main, side };
    }

    if (sim.checkpoint === "500k") {
      const hoverslamAltitude = (sim.vy * sim.vy) / (2 * (MAIN_ENGINE_POWER - Math.abs(GRAVITY) * 0.9));
      let main = altitude <= hoverslamAltitude * 1.25 ? 0.95 : 0.05;
      let side = -sim.angle * 4.5 - sim.angularVel * 1.5;
      return { main: Math.max(0, Math.min(1, main)), side: Math.max(-1, Math.min(1, side)) };
    }

    // 1.5M and 3.0M (Master Policy): High-Performance Dynamic Intercept
    const timeToTouchdown = Math.max(0.4, Math.abs(altitude / Math.min(-1.0, sim.vy)));
    const predictedPadX = sim.padX + (sim.padDrift ? sim.padVx * timeToTouchdown * 0.85 : 0);
    const leadDx = sim.x - predictedPadX;

    let targetAngle = 0.0;
    if (altitude > 50.0) {
      targetAngle = Math.max(-0.24, Math.min(0.24, -leadDx * 0.003 - (sim.vx - sim.padVx) * 0.035));
    } else {
      targetAngle = Math.max(-0.05, Math.min(0.05, -leadDx * 0.002));
    }

    const angleErr = sim.angle - targetAngle;
    let side = -angleErr * 8.5 - sim.angularVel * 2.3;
    side = Math.max(-1.0, Math.min(1.0, side));

    // Suicide Burn Altitude: s = v^2 / 2a
    const availableDecel = MAIN_ENGINE_POWER - Math.abs(GRAVITY);
    const criticalDecelAlt = (sim.vy * sim.vy) / (2.0 * availableDecel * 0.93);

    let main = 0.0;
    if (altitude <= criticalDecelAlt + 10.0) {
      const targetVy = -Math.min(26.0, Math.sqrt(Math.max(0.5, 2.0 * availableDecel * Math.max(0.0, altitude - 2.0))));
      const vyErr = targetVy - sim.vy;
      main = (Math.abs(GRAVITY) + vyErr * 3.6) / MAIN_ENGINE_POWER;
      main = Math.max(0.15, Math.min(1.0, main));
    } else if (altitude < 100.0 && sim.vy < -22.0) {
      main = 0.45;
    }

    if (altitude < 8.0) {
      if (sim.vy < -1.8) main = 0.85;
      else main = Math.abs(GRAVITY) / MAIN_ENGINE_POWER;
    }

    return { main, side };
  }

  function stepPhysics() {
    if (sim.screenShake > 0) {
      sim.screenShake -= DT * 20.0;
      sim.shakeX = (Math.random() - 0.5) * sim.screenShake * 1.5;
      sim.shakeY = (Math.random() - 0.5) * sim.screenShake * 1.5;
    } else {
      sim.shakeX = 0;
      sim.shakeY = 0;
    }

    if (sim.padDrift && !sim.grounded) {
      sim.padX += sim.padVx * DT;
      const center = WORLD_W / 2.0;
      if (sim.padX > center + PAD_MOVE_RANGE) {
        sim.padX = center + PAD_MOVE_RANGE;
        sim.padDir = -1.0;
        sim.padVx = sim.padDir * PAD_MOVE_SPEED;
      } else if (sim.padX < center - PAD_MOVE_RANGE) {
        sim.padX = center - PAD_MOVE_RANGE;
        sim.padDir = 1.0;
        sim.padVx = sim.padDir * PAD_MOVE_SPEED;
      }
    }

    if (sim.mode === "auto") {
      const controls = computeAutopilotControls();
      sim.mainThrottle = controls.main;
      sim.sideThrottle = controls.side;
    } else {
      sim.mainThrottle = keys.ArrowUp ? 1.0 : 0.0;
      sim.sideThrottle = (keys.ArrowRight ? 1.0 : 0.0) - (keys.ArrowLeft ? 1.0 : 0.0);
    }

    let windForce = 0.0;
    if (sim.windGusts) {
      windForce = Math.sin(Date.now() * 0.003) * 7.0;
    }

    if (!sim.grounded) {
      const thrust = sim.mainThrottle * MAIN_ENGINE_POWER;
      const ax = -thrust * Math.sin(sim.angle) + windForce;
      const ay = thrust * Math.cos(sim.angle) + GRAVITY;

      sim.vx = (sim.vx + ax * DT) * LINEAR_DAMPING;
      sim.vy = (sim.vy + ay * DT) * LINEAR_DAMPING;
      sim.x += sim.vx * DT;
      sim.y += sim.vy * DT;

      const torque = sim.sideThrottle * SIDE_ENGINE_POWER;
      sim.angularVel = (sim.angularVel + torque * DT) * ANGULAR_DAMPING;
      sim.angle += sim.angularVel * DT;

      if (sim.x < 30) { sim.x = 30; sim.vx = 0; }
      if (sim.x > WORLD_W - 30) { sim.x = WORLD_W - 30; sim.vx = 0; }

      checkTouchdown();
    } else {
      sim.outcomeHold += DT;
      if (sim.outcomeHold > 2.8) {
        resetRocket();
      }
    }

    spawnParticles();
    updateParticles();
    updateTelemetryHUD();
  }

  function checkTouchdown() {
    const footLeftX = sim.x - (LEG_SPAN / 2.0) * Math.cos(sim.angle) + FOOT_OFFSET * Math.sin(sim.angle);
    const footLeftY = sim.y - (LEG_SPAN / 2.0) * Math.sin(sim.angle) - FOOT_OFFSET * Math.cos(sim.angle);

    const footRightX = sim.x + (LEG_SPAN / 2.0) * Math.cos(sim.angle) + FOOT_OFFSET * Math.sin(sim.angle);
    const footRightY = sim.y + (LEG_SPAN / 2.0) * Math.sin(sim.angle) - FOOT_OFFSET * Math.cos(sim.angle);

    const padLeft = sim.padX - PAD_HALF_WIDTH;
    const padRight = sim.padX + PAD_HALF_WIDTH;

    const onPadLeft = footLeftX >= padLeft && footLeftX <= padRight;
    const onPadRight = footRightX >= padLeft && footRightX <= padRight;

    sim.legLeftContact = footLeftY <= PAD_HEIGHT && onPadLeft;
    sim.legRightContact = footRightY <= PAD_HEIGHT && onPadRight;

    if (footLeftY <= PAD_HEIGHT || footRightY <= PAD_HEIGHT || sim.y <= PAD_HEIGHT) {
      sim.grounded = true;
      const speed = Math.sqrt(sim.vx * sim.vx + sim.vy * sim.vy);
      const isUpright = Math.abs(sim.angle) <= TIP_ANGLE;
      const isOverDeck = sim.x >= padLeft && sim.x <= padRight;

      if (speed <= MAX_IMPACT_SPEED && isUpright && isOverDeck) {
        sim.landed = true;
        sim.crashed = false;
        sim.vy = 0;
        sim.vx = 0;
        sim.angularVel = 0;
        sim.angle *= 0.2;
        showVerdict(true, speed, sim.angle);
        spawnDustKickup(sim.x, PAD_HEIGHT, 40);
      } else {
        sim.crashed = true;
        sim.landed = false;
        sim.screenShake = 16.0;
        showVerdict(false, speed, sim.angle, !isOverDeck ? "MISSED DRONE SHIP" : (speed > MAX_IMPACT_SPEED ? "EXCESSIVE IMPACT VELOCITY" : "TOPPLE OVERSTABILITY"));
        spawnCrashFireball(sim.x, Math.max(PAD_HEIGHT, sim.y));
      }
    }
  }

  function spawnParticles() {
    if (sim.crashed) return;

    // Main Engine Exhaust: Electric Volt & White Sparks
    if (sim.mainThrottle > 0.05) {
      const nozzleX = sim.x + Math.sin(sim.angle) * (ROCKET_HEIGHT / 2.0);
      const nozzleY = sim.y - Math.cos(sim.angle) * (ROCKET_HEIGHT / 2.0);

      const count = Math.floor(sim.mainThrottle * 4 + 2);
      for (let i = 0; i < count; i++) {
        const spread = (Math.random() - 0.5) * 0.25;
        const speed = 40.0 + Math.random() * 80.0 * sim.mainThrottle;
        const pVx = sim.vx + Math.sin(-sim.angle + spread) * speed;
        const pVy = sim.vy - Math.cos(-sim.angle + spread) * speed;

        sim.particles.push({
          x: nozzleX,
          y: nozzleY,
          vx: pVx,
          vy: pVy,
          life: 0.3 + Math.random() * 0.2,
          maxLife: 0.5,
          size: 2.0 + Math.random() * 3.5,
          color: Math.random() < 0.4 ? "#d4ff00" : (Math.random() < 0.7 ? "#ffffff" : "#71717a")
        });
      }
    }

    // Cold-Gas RCS: Crisp White Mist
    if (Math.abs(sim.sideThrottle) > 0.05) {
      const side = sim.sideThrottle > 0 ? 1 : -1;
      const portX = sim.x + side * (ROCKET_WIDTH / 2.0);
      const portY = sim.y + (ROCKET_HEIGHT / 4.0);

      for (let i = 0; i < 2; i++) {
        sim.particles.push({
          x: portX,
          y: portY,
          vx: sim.vx - side * (20 + Math.random() * 30),
          vy: sim.vy + (Math.random() - 0.5) * 8,
          life: 0.18 + Math.random() * 0.12,
          maxLife: 0.3,
          size: 1.5 + Math.random() * 1.5,
          color: "#ffffff"
        });
      }
    }
  }

  function spawnDustKickup(x, y, count) {
    for (let i = 0; i < count; i++) {
      const ang = Math.random() * Math.PI;
      const spd = 15.0 + Math.random() * 50.0;
      sim.particles.push({
        x: x + (Math.random() - 0.5) * 35,
        y: y,
        vx: Math.cos(ang) * spd,
        vy: Math.sin(ang) * spd,
        life: 0.5 + Math.random() * 0.7,
        maxLife: 1.2,
        size: 2.5 + Math.random() * 4.0,
        color: "#52525b"
      });
    }
  }

  function spawnCrashFireball(x, y) {
    for (let i = 0; i < 65; i++) {
      const ang = Math.random() * Math.PI * 2;
      const spd = 30.0 + Math.random() * 130.0;
      sim.particles.push({
        x: x,
        y: y,
        vx: Math.cos(ang) * spd,
        vy: Math.sin(ang) * spd,
        life: 0.4 + Math.random() * 0.8,
        maxLife: 1.2,
        size: 2.5 + Math.random() * 5.0,
        color: Math.random() < 0.4 ? "#ffffff" : (Math.random() < 0.8 ? "#d4ff00" : "#27272a")
      });
    }
  }

  function updateParticles() {
    const alive = [];
    for (const p of sim.particles) {
      p.x += p.vx * DT;
      p.y += p.vy * DT;
      p.life -= DT;
      if (p.life > 0) alive.push(p);
    }
    sim.particles = alive;
  }

  function toScreen(x, y) {
    return {
      x: x + sim.shakeX,
      y: (WORLD_H - y) + sim.shakeY
    };
  }

  function render() {
    ctx.clearRect(0, 0, WORLD_W, WORLD_H);

    // Deep Void Black Sky
    ctx.fillStyle = "#050507";
    ctx.fillRect(0, 0, WORLD_W, WORLD_H);

    // Subtle Stars
    for (const s of sim.stars) {
      ctx.fillStyle = `rgba(255, 255, 255, ${s.alpha})`;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Ocean Waterline
    const oceanLevel = toScreen(0, 0).y;
    ctx.fillStyle = "#0c0d12";
    ctx.fillRect(0, oceanLevel, WORLD_W, WORLD_H - oceanLevel);
    ctx.strokeStyle = "#27272a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, oceanLevel);
    ctx.lineTo(WORLD_W, oceanLevel);
    ctx.stroke();

    renderDroneShip();
    renderParticles();
    renderRocket();
    renderCrosshairs();
  }

  function renderDroneShip() {
    const padTop = toScreen(sim.padX, PAD_HEIGHT);
    const padW = PAD_HALF_WIDTH * 2;
    const deckH = 14;

    // Drone Ship Deck
    ctx.fillStyle = "#12131a";
    ctx.fillRect(padTop.x - PAD_HALF_WIDTH, padTop.y, padW, deckH);
    ctx.strokeStyle = "#3f3f46";
    ctx.lineWidth = 1;
    ctx.strokeRect(padTop.x - PAD_HALF_WIDTH, padTop.y, padW, deckH);

    // Electric Volt Landing Target & Crosshair
    ctx.strokeStyle = "rgba(212, 255, 0, 0.4)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(padTop.x, padTop.y + 7, 22, 0, Math.PI * 2);
    ctx.stroke();

    // Center Cross
    ctx.strokeStyle = "#d4ff00";
    ctx.beginPath();
    ctx.moveTo(padTop.x - 6, padTop.y + 7);
    ctx.lineTo(padTop.x + 6, padTop.y + 7);
    ctx.moveTo(padTop.x, padTop.y + 1);
    ctx.lineTo(padTop.x, padTop.y + 13);
    ctx.stroke();

    // Support Pylons
    ctx.fillStyle = "#18181b";
    ctx.fillRect(padTop.x - PAD_HALF_WIDTH + 8, padTop.y + deckH, 8, WORLD_H - padTop.y);
    ctx.fillRect(padTop.x + PAD_HALF_WIDTH - 16, padTop.y + deckH, 8, WORLD_H - padTop.y);
  }

  function renderRocket() {
    if (sim.crashed && sim.outcomeHold > 0.4) {
      const pos = toScreen(sim.x, PAD_HEIGHT);
      ctx.fillStyle = "#27272a";
      ctx.fillRect(pos.x - 14, pos.y - 6, 28, 6);
      return;
    }

    const pos = toScreen(sim.x, sim.y);

    ctx.save();
    ctx.translate(pos.x, pos.y);
    ctx.rotate(sim.angle);

    const halfW = ROCKET_WIDTH / 2.0;
    const halfH = ROCKET_HEIGHT / 2.0;

    // Rocket Body (Stark White Titanium)
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-halfW, -halfH, ROCKET_WIDTH, ROCKET_HEIGHT);
    ctx.strokeStyle = "#27272a";
    ctx.lineWidth = 1;
    ctx.strokeRect(-halfW, -halfH, ROCKET_WIDTH, ROCKET_HEIGHT);

    // Carbon Nosecone
    ctx.fillStyle = "#ededed";
    ctx.beginPath();
    ctx.moveTo(-halfW, -halfH);
    ctx.lineTo(0, -halfH - 8);
    ctx.lineTo(halfW, -halfH);
    ctx.closePath();
    ctx.fill();

    // Matte Black Interstage Band with Electric Volt Trim
    ctx.fillStyle = "#0c0d12";
    ctx.fillRect(-halfW, -halfH + 4, ROCKET_WIDTH, 5);
    ctx.fillStyle = "#d4ff00";
    ctx.fillRect(-halfW, -halfH + 9, ROCKET_WIDTH, 1.5);

    // Engine Nozzle
    ctx.fillStyle = "#27272a";
    ctx.beginPath();
    ctx.moveTo(-halfW + 3, halfH);
    ctx.lineTo(-halfW + 1, halfH + 6);
    ctx.lineTo(halfW - 1, halfH + 6);
    ctx.lineTo(halfW - 3, halfH);
    ctx.closePath();
    ctx.fill();

    // Landing Legs
    ctx.strokeStyle = "#a1a1aa";
    ctx.lineWidth = 2.5;

    ctx.beginPath();
    ctx.moveTo(-halfW + 1, halfH - 4);
    ctx.lineTo(-LEG_SPAN / 2.0, halfH + FOOT_DROP);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(halfW - 1, halfH - 4);
    ctx.lineTo(LEG_SPAN / 2.0, halfH + FOOT_DROP);
    ctx.stroke();

    ctx.restore();
  }

  function renderParticles() {
    for (const p of sim.particles) {
      const pos = toScreen(p.x, p.y);
      const alpha = p.life / p.maxLife;
      ctx.fillStyle = p.color;
      ctx.globalAlpha = Math.max(0, Math.min(1, alpha));
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, p.size, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1.0;
  }

  function renderCrosshairs() {
    const padTop = toScreen(sim.padX, PAD_HEIGHT);
    ctx.strokeStyle = "rgba(212, 255, 0, 0.2)";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);

    ctx.beginPath();
    ctx.moveTo(padTop.x, 0);
    ctx.lineTo(padTop.x, WORLD_H);
    ctx.stroke();

    ctx.setLineDash([]);
  }

  function updateTelemetryHUD() {
    const elAlt = document.getElementById("hud-alt");
    const elVy = document.getElementById("hud-vy");
    const elVx = document.getElementById("hud-vx");
    const elPitch = document.getElementById("hud-pitch");
    const elThrottleBar = document.getElementById("hud-throttle-bar");
    const elThrottleVal = document.getElementById("hud-throttle-val");
    const elLegL = document.getElementById("hud-leg-left");
    const elLegR = document.getElementById("hud-leg-right");
    const elPadVx = document.getElementById("hud-pad-vx");

    const alt = Math.max(0, sim.y - PAD_HEIGHT);
    if (elAlt) elAlt.textContent = `${alt.toFixed(1)} m`;
    if (elVy) {
      elVy.textContent = `${sim.vy >= 0 ? "+" : ""}${sim.vy.toFixed(1)} m/s`;
      elVy.className = sim.vy < -25 ? "text-white font-bold" : (sim.vy < -12 ? "text-zinc-400 font-bold" : "text-volt font-bold");
    }
    if (elVx) elVx.textContent = `${sim.vx >= 0 ? "+" : ""}${sim.vx.toFixed(1)} m/s`;

    const pitchDeg = (sim.angle * 180 / Math.PI).toFixed(1);
    if (elPitch) elPitch.textContent = `${pitchDeg}°`;

    const throttlePct = Math.round(sim.mainThrottle * 100);
    if (elThrottleBar) elThrottleBar.style.width = `${throttlePct}%`;
    if (elThrottleVal) elThrottleVal.textContent = `${throttlePct}%`;

    if (elLegL) elLegL.style.backgroundColor = sim.legLeftContact ? "#d4ff00" : "#27272a";
    if (elLegR) elLegR.style.backgroundColor = sim.legRightContact ? "#d4ff00" : "#27272a";
    if (elPadVx) elPadVx.textContent = `${sim.padDrift ? Math.abs(sim.padVx).toFixed(1) : "0.0"} m/s`;
  }

  function showVerdict(success, speed, angle, reason) {
    const elVerdict = document.getElementById("hud-verdict");
    const elTitle = document.getElementById("hud-verdict-title");
    const elSub = document.getElementById("hud-verdict-sub");
    if (!elVerdict || !elTitle || !elSub) return;

    elVerdict.classList.remove("hidden");
    if (success) {
      elVerdict.className = "mx-auto py-2 px-6 rounded border text-center backdrop-blur-md bg-black/80 border-volt text-volt shadow-volt";
      elTitle.textContent = "NOMINAL TOUCHDOWN // HOVERSLAM LOCKED";
      elSub.textContent = `Sink Rate: ${speed.toFixed(1)} m/s | Tilt: ${(angle * 180 / Math.PI).toFixed(1)}° | Quality: +148.5`;
    } else {
      elVerdict.className = "mx-auto py-2 px-6 rounded border text-center backdrop-blur-md bg-black/80 border-white/40 text-white";
      elTitle.textContent = "MISSION ABORT // STRUCTURAL OVERLOAD";
      elSub.textContent = reason || `Impact: ${speed.toFixed(1)} m/s (Limit: ${MAX_IMPACT_SPEED} m/s)`;
    }
  }

  function hideVerdict() {
    const elVerdict = document.getElementById("hud-verdict");
    if (elVerdict) elVerdict.classList.add("hidden");
  }

  function loop() {
    stepPhysics();
    render();
    requestAnimationFrame(loop);
  }

  resetRocket();
  requestAnimationFrame(loop);

  window.HoverslamSim = {
    reset: resetRocket,
    setMode: (mode) => {
      sim.mode = mode;
      const hint = document.getElementById("manual-controls-hint");
      if (hint) {
        hint.classList.toggle("hidden", mode !== "manual");
      }
    },
    setCheckpoint: (ckpt) => {
      sim.checkpoint = ckpt;
      resetRocket();
    },
    togglePadDrift: () => {
      sim.padDrift = !sim.padDrift;
      return sim.padDrift;
    },
    toggleWindGusts: () => {
      sim.windGusts = !sim.windGusts;
      return sim.windGusts;
    }
  };

})();
