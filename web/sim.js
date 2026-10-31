/**
 * HOVERSLAM // Real-time In-Browser Kinematic Physics & Autopilot Engine
 * Aesthetic: Silent, Mathematical, Exact
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
    mode: "auto",
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

    // Status
    landed: false,
    crashed: false,
    grounded: false,
    outcomeHold: 0,
    screenShake: 0,
    shakeX: 0,
    shakeY: 0,

    // Gear
    legLeftContact: false,
    legRightContact: false,

    particles: [],
    stars: []
  };

  // Subtle starfield
  for (let i = 0; i < 75; i++) {
    sim.stars.push({
      x: Math.random() * WORLD_W,
      y: Math.random() * (WORLD_H - 120),
      r: Math.random() * 1.0 + 0.4,
      alpha: Math.random() * 0.4 + 0.15
    });
  }

  // Keyboard controls for manual pilot
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
    sim.x = 240.0 + Math.random() * 320.0;
    sim.y = 490.0 + Math.random() * 60.0;
    sim.vx = (Math.random() - 0.5) * 3.5;
    sim.vy = -Math.random() * 4.0;
    sim.angle = (Math.random() - 0.5) * 0.10;
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

    sim.padX = WORLD_W / 2.0 + (Math.random() - 0.5) * 90.0;
    sim.padDir = Math.random() < 0.5 ? 1.0 : -1.0;
    sim.padVx = sim.padDir * PAD_MOVE_SPEED;

    hideVerdict();
  }

  function computeAutopilotControls() {
    if (sim.landed || sim.crashed) {
      return { main: 0.0, side: 0.0 };
    }

    // Altitude measured from bottom of landing gear feet to the drone ship deck surface
    const altitude = Math.max(0.0, sim.y - FOOT_OFFSET - PAD_HEIGHT);
    const targetX = sim.padX;

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

    // 1.5M and 3.0M (Master Policy)
    // Continuous trajectory guidance: keep booster on glideslope toward moving deck
    const dx = sim.x - sim.padX;
    const targetVx = sim.padDrift ? sim.padVx : 0.0;
    const dvx = sim.vx - targetVx;

    let targetAngle = 0.0;
    if (altitude > 30.0) {
      // Tilt right (positive angle) when dx < 0 to steer right; tilt left when dx > 0
      targetAngle = Math.max(-0.28, Math.min(0.28, -dx * 0.0035 - dvx * 0.040));
    } else {
      // Final 30m: enforce vertical touchdown alignment while matching deck drift speed
      const blend = Math.max(0.0, (altitude - 4.0) / 26.0);
      targetAngle = Math.max(-0.04, Math.min(0.04, -dx * 0.001 - dvx * 0.025)) * blend;
    }

    const angleErr = sim.angle - targetAngle;
    let side = -angleErr * 9.5 - sim.angularVel * 2.8;
    side = Math.max(-1.0, Math.min(1.0, side));

    // Controlled descent rate & final hoverslam deceleration
    const availableDecel = MAIN_ENGINE_POWER - Math.abs(GRAVITY); // 81 m/s^2
    let desiredVy = -32.0;

    if (altitude < 140.0) {
      // Suicide burn braking curve converging to -1.0 m/s at altitude 0
      desiredVy = -Math.max(1.0, Math.sqrt(2.0 * availableDecel * 0.85 * Math.max(0.1, altitude)));
      desiredVy = Math.max(-32.0, desiredVy);
    }

    if (altitude < 8.0) {
      desiredVy = -1.0;
    }

    const vyErr = desiredVy - sim.vy;
    let main = (Math.abs(GRAVITY) + vyErr * 4.5) / MAIN_ENGINE_POWER;
    main = Math.max(0.10, Math.min(1.0, main));

    if (altitude < 6.0) {
      if (sim.vy < -1.8) main = 0.95;
      else if (sim.vy > -0.3) main = 0.15;
      else main = (Math.abs(GRAVITY) - 1.0) / MAIN_ENGINE_POWER;
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
      windForce = Math.sin(Date.now() * 0.003) * 6.5;
    }

    if (!sim.grounded) {
      const thrust = sim.mainThrottle * MAIN_ENGINE_POWER;
      const ax = thrust * Math.sin(sim.angle) + windForce;
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

    if (footLeftY <= PAD_HEIGHT || footRightY <= PAD_HEIGHT || (sim.y - FOOT_OFFSET) <= PAD_HEIGHT) {
      sim.grounded = true;
      const targetPadVx = sim.padDrift ? sim.padVx : 0.0;
      const relVx = sim.vx - targetPadVx;
      const speed = Math.sqrt(relVx * relVx + sim.vy * sim.vy);
      const isUpright = Math.abs(sim.angle) <= TIP_ANGLE;
      const isOverDeck = sim.x >= padLeft && sim.x <= padRight;

      if (speed <= MAX_IMPACT_SPEED && isUpright && isOverDeck) {
        sim.landed = true;
        sim.crashed = false;
        sim.vy = 0;
        sim.vx = targetPadVx;
        sim.angularVel = 0;
        sim.angle *= 0.15;
        showVerdict(true, speed, sim.angle);
        spawnDustKickup(sim.x, PAD_HEIGHT, 35);
      } else {
        sim.crashed = true;
        sim.landed = false;
        sim.screenShake = 14.0;
        showVerdict(false, speed, sim.angle, !isOverDeck ? "Off-deck contact" : (speed > MAX_IMPACT_SPEED ? "Impact velocity overload" : "Topple instability"));
        spawnCrashFireball(sim.x, Math.max(PAD_HEIGHT, sim.y));
      }
    }
  }

  function spawnParticles() {
    if (sim.crashed) return;

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
          color: Math.random() < 0.6 ? "#ededed" : "#71717a"
        });
      }
    }

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
          color: "#a1a1aa"
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
    for (let i = 0; i < 60; i++) {
      const ang = Math.random() * Math.PI * 2;
      const spd = 30.0 + Math.random() * 120.0;
      sim.particles.push({
        x: x,
        y: y,
        vx: Math.cos(ang) * spd,
        vy: Math.sin(ang) * spd,
        life: 0.4 + Math.random() * 0.8,
        maxLife: 1.2,
        size: 2.5 + Math.random() * 4.5,
        color: Math.random() < 0.6 ? "#ededed" : "#3f3f46"
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

    // Void background
    ctx.fillStyle = "#050507";
    ctx.fillRect(0, 0, WORLD_W, WORLD_H);

    // Faint stars
    for (const s of sim.stars) {
      ctx.fillStyle = `rgba(255, 255, 255, ${s.alpha})`;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fill();
    }

    // Waterline
    const oceanLevel = toScreen(0, 0).y;
    ctx.fillStyle = "#08090d";
    ctx.fillRect(0, oceanLevel, WORLD_W, WORLD_H - oceanLevel);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
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

    ctx.fillStyle = "#0f1118";
    ctx.fillRect(padTop.x - PAD_HALF_WIDTH, padTop.y, padW, deckH);
    ctx.strokeStyle = "rgba(255, 255, 255, 0.15)";
    ctx.lineWidth = 1;
    ctx.strokeRect(padTop.x - PAD_HALF_WIDTH, padTop.y, padW, deckH);

    // Subtle landing crosshair
    ctx.strokeStyle = "rgba(255, 255, 255, 0.35)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(padTop.x, padTop.y + 7, 20, 0, Math.PI * 2);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(padTop.x - 5, padTop.y + 7);
    ctx.lineTo(padTop.x + 5, padTop.y + 7);
    ctx.moveTo(padTop.x, padTop.y + 2);
    ctx.lineTo(padTop.x, padTop.y + 12);
    ctx.stroke();

    ctx.fillStyle = "#12141d";
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

    // Body
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(-halfW, -halfH, ROCKET_WIDTH, ROCKET_HEIGHT);
    ctx.strokeStyle = "#27272a";
    ctx.lineWidth = 1;
    ctx.strokeRect(-halfW, -halfH, ROCKET_WIDTH, ROCKET_HEIGHT);

    // Nosecone
    ctx.fillStyle = "#ededed";
    ctx.beginPath();
    ctx.moveTo(-halfW, -halfH);
    ctx.lineTo(0, -halfH - 8);
    ctx.lineTo(halfW, -halfH);
    ctx.closePath();
    ctx.fill();

    // Dark interstage band
    ctx.fillStyle = "#18181b";
    ctx.fillRect(-halfW, -halfH + 4, ROCKET_WIDTH, 5);

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
    ctx.lineWidth = 2.0;

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
    ctx.strokeStyle = "rgba(255, 255, 255, 0.08)";
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
    const elPadVx = document.getElementById("hud-pad-vx");

    const alt = Math.max(0, sim.y - FOOT_OFFSET - PAD_HEIGHT);
    if (elAlt) elAlt.textContent = `${alt.toFixed(1)} m`;
    if (elVy) elVy.textContent = `${sim.vy >= 0 ? "+" : ""}${sim.vy.toFixed(1)} m/s`;
    if (elVx) elVx.textContent = `${sim.vx >= 0 ? "+" : ""}${sim.vx.toFixed(1)} m/s`;

    const pitchDeg = (sim.angle * 180 / Math.PI).toFixed(1);
    if (elPitch) elPitch.textContent = `${pitchDeg}°`;

    const throttlePct = Math.round(sim.mainThrottle * 100);
    if (elThrottleBar) elThrottleBar.style.width = `${throttlePct}%`;
    if (elThrottleVal) elThrottleVal.textContent = `${throttlePct}%`;

    if (elPadVx) elPadVx.textContent = `${sim.padDrift ? Math.abs(sim.padVx).toFixed(1) : "0.0"} m/s`;
  }

  function showVerdict(success, speed, angle, reason) {
    const elVerdict = document.getElementById("hud-verdict");
    const elTitle = document.getElementById("hud-verdict-title");
    const elSub = document.getElementById("hud-verdict-sub");
    if (!elVerdict || !elTitle || !elSub) return;

    elVerdict.classList.remove("hidden");
    if (success) {
      elVerdict.className = "mx-auto py-2 px-5 rounded-full border text-center backdrop-blur-md bg-black/80 border-white/20 text-white";
      elTitle.textContent = "Touchdown confirmed";
      elSub.textContent = `${speed.toFixed(1)} m/s at touch · ${(angle * 180 / Math.PI).toFixed(1)}° pitch`;
    } else {
      elVerdict.className = "mx-auto py-2 px-5 rounded-full border text-center backdrop-blur-md bg-black/80 border-zinc-700 text-zinc-300";
      elTitle.textContent = "Flight terminated";
      elSub.textContent = reason || `${speed.toFixed(1)} m/s impact`;
    }
  }

  function hideVerdict() {
    const elVerdict = document.getElementById("hud-verdict");
    if (elVerdict) elVerdict.classList.add("hidden");
  }

  let isSimVisible = true;
  let animId = null;

  function loop() {
    if (!isSimVisible || document.hidden) {
      animId = null;
      return;
    }
    stepPhysics();
    render();
    animId = requestAnimationFrame(loop);
  }

  function startLoop() {
    if (!animId && isSimVisible && !document.hidden) {
      animId = requestAnimationFrame(loop);
    }
  }

  // Optimize CPU/battery: pause physics/rendering when canvas is off-screen
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        isSimVisible = entry.isIntersecting;
        if (isSimVisible) startLoop();
      });
    }, { threshold: 0.05 });
    observer.observe(canvas);
  }

  document.addEventListener("visibilitychange", () => {
    if (!document.hidden && isSimVisible) startLoop();
  });

  resetRocket();
  startLoop();

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
