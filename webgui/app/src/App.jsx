// Cellvision web GUI — single SPA driven by /api/state.
//
// Layers (clearly separated below):
//   1. API client          — fetch wrappers, no UI knowledge
//   2. Geometry            — pure rotation/scaling math
//   3. Color helpers       — pure
//   4. Stage components    — ImageStage, MeaStage  (presentation only)
//   5. Views               — VerificationView, SelectionView  (orchestrate state + stages)
//   6. App + bootstrap

import {
  Fragment,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";


// ──────────────────────────────────────────────────────────────────────────────
// 1. API client
// ──────────────────────────────────────────────────────────────────────────────

async function apiFetchState() {
  const r = await fetch("/api/state");
  if (!r.ok) throw new Error(`/api/state: ${r.status}`);
  return r.json();
}

async function apiDispatch(name, payload = {}) {
  const r = await fetch("/api/event", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name, payload }),
  });
  if (!r.ok) throw new Error(`${name}: ${r.status} ${await r.text()}`);
  return r.json();
}

async function apiRestart() {
  const r = await fetch("/api/restart", { method: "POST" });
  if (!r.ok) throw new Error(`/api/restart: ${r.status}`);
  return r.json();
}

// ──────────────────────────────────────────────────────────────────────────────
// 2. Geometry — image-space coordinate system, rotated frame, display pixels.
// ──────────────────────────────────────────────────────────────────────────────

function rotPoint(x, y, rot, W, H) {
  if (rot === 0) return [x, y];
  if (rot === 90) return [H - y, x];
  if (rot === 180) return [W - x, H - y];
  return [y, W - x]; // 270
}

function invRotPoint(rx, ry, rot, W, H) {
  if (rot === 0) return [rx, ry];
  if (rot === 90) return [ry, H - rx];
  if (rot === 180) return [W - rx, H - ry];
  return [W - ry, rx]; // 270
}

function rotatedDims(rot, W, H) {
  return rot % 180 === 0 ? [W, H] : [H, W];
}

// CSS transform for the <img> so it lands at (0,0)..(rotW,rotH) before scaling.
function imgTransform(rot, W, H, scale) {
  const t = { 0: [0, 0], 90: [H, 0], 180: [W, H], 270: [0, W] }[rot];
  return `scale(${scale}) translate(${t[0]}px, ${t[1]}px) rotate(${rot}deg)`;
}

// ──────────────────────────────────────────────────────────────────────────────
// 3. Colors
// ──────────────────────────────────────────────────────────────────────────────

function makeColors(n) {
  return Array.from(
    { length: n },
    (_, i) => `hsl(${Math.round((i / n) * 360)}, 70%, 55%)`,
  );
}

// ──────────────────────────────────────────────────────────────────────────────
// 4. Stage components — pure presentation, parent passes everything via props.
// ──────────────────────────────────────────────────────────────────────────────

const STAGE_PX = 720;
const BTN_R = 14;

/**
 * Image stage: rotated image, buttons over each detected box, optional
 * markers/decorations, and an underlying click handler for the bare image.
 *
 * Props:
 *   imageSize: [W, H]
 *   rot: 0 | 90 | 180 | 270
 *   boxes: [{cx, cy, color, label, onClick, dim}]  (rendered as buttons)
 *   markers: [{x, y, kind, label, color}]          (rendered as decorations)
 *   onImageClick(x, y): called with original image coords; null disables it
 */
function ImageStage({ imageSize, rot, boxes, markers, polys, onImageClick }) {
  const [W, H] = imageSize;
  const [rotW, rotH] = rotatedDims(rot, W, H);
  const scale = Math.min(STAGE_PX / rotW, STAGE_PX / rotH);
  const dispW = rotW * scale;
  const dispH = rotH * scale;

  const handleClick = (e) => {
    if (!onImageClick) return;
    if (e.target !== e.currentTarget) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const rx = (e.clientX - rect.left) / scale;
    const ry = (e.clientY - rect.top) / scale;
    if (rx < 0 || ry < 0 || rx > rotW || ry > rotH) return;
    const [ox, oy] = invRotPoint(rx, ry, rot, W, H);
    onImageClick(ox, oy);
  };

  const place = (x, y) => {
    const [rx, ry] = rotPoint(x, y, rot, W, H);
    return { left: rx * scale, top: ry * scale };
  };

  return <div
      class="relative bg-black stage-clip select-none"
      style={{
        width: dispW,
        height: dispH,
        cursor: onImageClick ? "crosshair" : "default",
      }}
      onClick={handleClick}
    >
      <img
        src="/api/image"
        alt="image microscope"
        draggable="false"
        class="absolute top-0 left-0 pointer-events-none"
        style={{
          width: `${W}px`,
          height: `${H}px`,
          maxWidth: "none",
          maxHeight: "none",
          transformOrigin: "0 0",
          transform: imgTransform(rot, W, H, scale),
        }}
      />

      {polys && polys.length
        ? <svg
            class="absolute top-0 left-0 pointer-events-none"
            width={dispW}
            height={dispH}
            viewBox={`0 0 ${dispW} ${dispH}`}
          >
            {polys.map((p, i) => {
              const pts = p.points
                .map(([x, y]) => {
                  const { left, top } = place(x, y);
                  return `${left},${top}`;
                })
                .join(" ");
              return <polygon
                key={`p-${i}`}
                points={pts}
                fill={p.fill ?? "rgba(255,255,255,0.12)"}
                stroke={p.stroke ?? "#ffffff"}
                stroke-width={p.strokeWidth ?? 2}
              />;
            })}
          </svg>
        : null}

      {markers.map(
        (m, i) =>
          <Marker key={`m-${i}`} {...m} place={place} rot={rot} />,
      )}

      {boxes.map(
        (b, i) =>
          <BoxButton key={`b-${b.boxIdx ?? i}`} {...b} place={place} />,
      )}
    </div>
  ;
}

function BoxButton({ cx, cy, color, label, dim, onClick, place, ring }) {
  const { left, top } = place(cx, cy);
  return <button
      type="button"
      onClick={(e) => {
        e.stopPropagation();
        onClick?.();
      }}
      class={[
        "absolute rounded-full border-2 transition",
        onClick ? "cursor-pointer hover:scale-110" : "cursor-default",
        ring ? "ring-4 ring-amber-400/60" : "",
      ].join(" ")}
      style={{
        left: left - BTN_R,
        top: top - BTN_R,
        width: 2 * BTN_R,
        height: 2 * BTN_R,
        background: color,
        opacity: dim ? 0.55 : 1,
        borderColor: "white",
      }}
    >
      {label
        ? <span
            class="absolute -top-4 left-1/2 -translate-x-1/2 text-[10px] font-bold text-white whitespace-nowrap drop-shadow"
            style={{ textShadow: "0 1px 2px rgba(0,0,0,.9)" }}
          >
            {label}
          </span>
        : null}
    </button>
  ;
}

function Marker({ kind, x, y, color, label, place }) {
  const { left, top } = place(x, y);
  const common = {
    position: "absolute",
    left: left - BTN_R,
    top: top - BTN_R,
    width: 2 * BTN_R,
    height: 2 * BTN_R,
    pointerEvents: "none",
  };
  if (kind === "hint") {
    return <div
        class="rounded-full"
        style={{
          ...common,
          border: `2px dashed ${color}`,
        }}
      ></div>
    ;
  }
  if (kind === "hint-current") {
    return <div
        class="rounded-full animate-pulse"
        style={{
          ...common,
          border: `3px dashed ${color}`,
          left: left - BTN_R - 3,
          top: top - BTN_R - 3,
          width: 2 * BTN_R + 6,
          height: 2 * BTN_R + 6,
        }}
      >
        <div
          class="absolute -top-4 left-1/2 -translate-x-1/2 text-[10px] font-bold whitespace-nowrap"
          style={{ color, textShadow: "0 1px 2px rgba(0,0,0,.9)" }}
        >
          → {label}
        </div>
      </div>
    ;
  }
  if (kind === "x") {
    return <div style={common}>
        <svg viewBox="0 0 ${2 * BTN_R} ${2 * BTN_R}" width={2 * BTN_R} height={2 * BTN_R}>
          <line x1={4} y1={4} x2={2 * BTN_R - 4} y2={2 * BTN_R - 4}
            stroke={color} stroke-width="3" />
          <line x1={2 * BTN_R - 4} y1={4} x2={4} y2={2 * BTN_R - 4}
            stroke={color} stroke-width="3" />
        </svg>
        {label
          ? <span
              class="absolute -top-4 left-1/2 -translate-x-1/2 text-[10px] font-bold"
              style={{ color, textShadow: "0 1px 2px rgba(0,0,0,.9)" }}
            >
              {label}
            </span>
          : null}
      </div>
    ;
  }
  return null;
}

/**
 * MEA schema panel. Each pattern point is a button positioned in the
 * bounding box of the pattern, scaled to fit MEA_PX.
 *
 * Props:
 *   patternPoints, patternLabels
 *   pointStyleFor(idx) -> {color, size, ring, label?, onClick?}
 */
const MEA_PX = 540;
const MEA_PAD = 30;

function MeaStage({ patternPoints, pointStyleFor }) {
  const { ox, oy, sc } = useMemo(() => {
    let mnx = Infinity, mny = Infinity, mxx = -Infinity, mxy = -Infinity;
    for (const [x, y] of patternPoints) {
      if (x < mnx) mnx = x;
      if (y < mny) mny = y;
      if (x > mxx) mxx = x;
      if (y > mxy) mxy = y;
    }
    const rngX = Math.max(mxx - mnx, 1e-9);
    const rngY = Math.max(mxy - mny, 1e-9);
    const usable = MEA_PX - 2 * MEA_PAD;
    const sc = Math.min(usable / rngX, usable / rngY);
    const ox = (MEA_PX - rngX * sc) / 2 - mnx * sc;
    const oy = (MEA_PX - rngY * sc) / 2 - mny * sc;
    return { ox, oy, sc };
  }, [patternPoints]);

  return <div
      class="relative bg-white text-slate-900 rounded-md select-none"
      style={{ width: MEA_PX, height: MEA_PX }}
    >
      {patternPoints.map((p, i) => {
        const st = pointStyleFor(i);
        const cx = p[0] * sc + ox;
        const cy = p[1] * sc + oy;
        const r = st.size ?? 6;
        return <button
          key={i}
          type="button"
          onClick={st.onClick ?? null}
          class={[
            "absolute rounded-full border border-white",
            st.onClick ? "cursor-pointer" : "cursor-default",
            st.ring ? "ring-2 ring-amber-500/70" : "",
          ].join(" ")}
          style={{
            left: cx - r,
            top: cy - r,
            width: 2 * r,
            height: 2 * r,
            background: st.color,
          }}
        >
          {st.label
            ? <span
                class="absolute -top-3 left-1/2 -translate-x-1/2 text-[9px] font-bold whitespace-nowrap text-slate-700"
              >
                {st.label}
              </span>
            : null}
        </button>;
      })}
    </div>
  ;
}

// ──────────────────────────────────────────────────────────────────────────────
// 5. Views — wire state + dispatch into the stage components.
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Trigger a session-wide restart. Confirms first, then transitions
 * local state to "idle" so the polling effect reactivates and picks
 * up the next workflow generation.
 */
function RestartButton({ onRestart, variant = "header" }) {
  const cls =
    variant === "header"
      ? "fixed top-3 right-3 z-50 px-3 py-1.5 rounded-md text-sm font-semibold text-white shadow bg-purple-600/95 hover:bg-purple-500 backdrop-blur"
      : "px-5 py-2 rounded-md font-semibold text-white shadow bg-purple-600 hover:bg-purple-500";
  return (
    <button type="button" onClick={onRestart} class={cls}>
      ↻ Recommencer
    </button>
  );
}

function PrimaryButton({ children, onClick, color = "emerald", disabled }) {
  const colorMap = {
  emerald: "bg-orange-500 hover:bg-orange-400",
  red: "bg-red-600 hover:bg-red-500",
  amber: "bg-purple-600 hover:bg-purple-500",
  slate: "bg-slate-500 hover:bg-slate-400",
};

  return <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      class={`px-5 py-2 rounded-md font-semibold text-white shadow disabled:opacity-40 disabled:cursor-not-allowed ${colorMap[color]}`}
    >
      {children}
    </button>
  ;
}

function SecondaryButton({ children, onClick, active, disabled }) {
  return <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      class={[
        "px-3 py-1.5 rounded-md text-sm font-medium transition",
        "border border-slate-300",
        active ? "bg-orange-500 text-white border-orange-500" : "bg-white hover:bg-slate-100 text-slate-700",
        disabled ? "opacity-40 cursor-not-allowed" : "",
      ].join(" ")}
    >
      {children}
    </button>
  ;
}

function StatusBar({ children, tone = "info" }) {
  const toneCls = {
    info: "bg-orange-100 text-orange-900 border-b border-orange-200",
    warn: "bg-purple-100 text-purple-900 border-b border-purple-200",
    err: "bg-red-100 text-red-900 border-b border-red-200",
  }[tone];
  return <div
    class={`w-full px-4 py-3 text-center font-semibold ${toneCls}`}
  >
    {children}
  </div>;
}

function RotateBar({ setRot, children }) {
  return <div class="flex items-center gap-2">
    <span class="text-sm text-slate-700">Image microscope</span>
    <SecondaryButton onClick={() => setRot((r) => (r + 270) % 360)}>↺ Anti-horaire</SecondaryButton>
    <SecondaryButton onClick={() => setRot((r) => (r + 90) % 360)}>↻ Horaire</SecondaryButton>
    {children}
  </div>;
}

// ── Verification view ────────────────────────────────────────────────────────

function VerificationView({ state, rot, setRot, dispatch }) {
  const labelPerBox = useMemo(() => {
    const m = {};
    for (const k of Object.keys(state.label_per_box)) {
      m[Number(k)] = state.label_per_box[k];
    }
    return m;
  }, [state.label_per_box]);

  const boxes = state.detected_boxes.map((b, i) => {
    const cx = (b[0] + b[2]) / 2;
    const cy = (b[1] + b[3]) / 2;
    const lbl = labelPerBox[i];
    return {
      boxIdx: i,
      cx,
      cy,
      color: lbl ? "#1abc9c" : "#7f8c8d",
      label: lbl ?? null,
      dim: !lbl,
    };
  });

  const markers = [];
  // Manually placed (already done)
  for (const [lbl, pos] of Object.entries(state.manually_placed)) {
    markers.push({ kind: "x", x: pos[0], y: pos[1], color: "#27ae60", label: lbl });
  }
  // Missing hints during placement phase
  if (state.phase === "missing") {
    for (const lbl of state.missing_labels) {
      if (state.manually_placed[lbl]) continue;
      const hint = state.missing_hint[lbl];
      if (!hint) continue;
      const isCurrent = lbl === state.current_missing_label;
      markers.push({
        kind: isCurrent ? "hint-current" : "hint",
        x: hint[0],
        y: hint[1],
        color: isCurrent ? "#e67e22" : "#f39c12",
        label: lbl,
      });
    }
  }

  const isMissing = state.phase === "missing";
  const labeledLabels = new Set(Object.values(labelPerBox));
  const placedLabels = new Set(Object.keys(state.manually_placed));

  const meaStyle = (i) => {
    const lbl = state.pattern_labels[i];
    const cur = state.current_missing_label;
    if (labeledLabels.has(lbl) || placedLabels.has(lbl))
      return { color: "#1abc9c", size: 5, label: lbl };
    if (lbl === cur)
      return { color: "#e67e22", size: 9, ring: true, label: lbl };
    if (state.missing_labels.includes(lbl))
      return {
        color: isMissing ? "#f39c12" : "#e67e22",
        size: 5,
        label: lbl,
      };
    return { color: "#bdc3c7", size: 3, label: lbl };
  };

  const onImageClick = isMissing
    ? (x, y) => dispatch("place_at", { x, y })
    : null;

  const status = isMissing
    ? `Placer '${state.current_missing_label}' (${state.missing_step + 1}/${state.missing_labels.length}) — cliquez sur l'image, Placer automatiquement, ou Passer`
    : (() => {
        const labeled = Object.keys(labelPerBox).length;
        const total = state.detected_boxes.length;
        const miss = state.missing_labels.length;
        let s = `${labeled} / ${total} électrodes détectées étiquetées`;
        if (miss) s += ` — ${miss} électrode${miss > 1 ? "s" : ""} MEA non trouvée${miss > 1 ? "s" : ""} (en orange)`;
        return s;
      })();

  return <div class="flex flex-col h-full">
      <StatusBar tone={isMissing ? "warn" : "info"}>{status}</StatusBar>

      <div class="flex-1 overflow-auto p-6 grid grid-cols-1 xl:grid-cols-2 gap-6 justify-items-center items-start">
        <div class="flex flex-col items-center gap-3">
          <RotateBar rot={rot} setRot={setRot} />
          <ImageStage
            imageSize={state.image_size}
            rot={rot}
            boxes={boxes}
            markers={markers}
            onImageClick={onImageClick}
          />
        </div>
        <div class="flex flex-col items-center gap-3">
          <span class="text-sm text-slate-700">MEA: {state.mea_family}</span>

          <MeaStage
            patternPoints={state.pattern_points}
            patternLabels={state.pattern_labels}
            pointStyleFor={meaStyle}
          />
        </div>
      </div>

      <div class="flex justify-center gap-4 pb-6">
        {state.phase === "review"
          ? <>
              <PrimaryButton color="emerald" onClick={() => dispatch("approve")}>
                Approuver — utiliser le résultat automatique
              </PrimaryButton>
              <PrimaryButton color="red" onClick={() => dispatch("reject")}>
                Re-labelliser manuellement
              </PrimaryButton>
        </> : <>
              <PrimaryButton color="amber" onClick={() => dispatch("autoplace")}>
                Placer automatiquement
              </PrimaryButton>
              <PrimaryButton color="slate" onClick={() => dispatch("skip")}>
                Passer (non visible)
              </PrimaryButton>
            </>}
      </div>
    </div>
  ;
}

// ── Selection view ───────────────────────────────────────────────────────────

function SelectionView({ state, rot, setRot, dispatch }) {
  const colors = useMemo(() => makeColors(state.n_landmarks), [state.n_landmarks]);
  const assignedBoxToStep = useMemo(() => {
    const m = {};
    state.selections.forEach((s, i) => {
      if (s.type === "box") m[s.box_idx] = i;
    });
    return m;
  }, [state.selections]);

  const inSelect = state.phase === "select";
  const inMissing = state.phase === "missing";
  const inVerify = state.phase === "verify";
  const stepDone = state.step >= state.n_landmarks;

  const boxLabels = useMemo(() => {
    const m = {};
    for (const k of Object.keys(state.box_labels)) m[Number(k)] = state.box_labels[k];
    return m;
  }, [state.box_labels]);

  const boxes = state.detected_boxes.map((b, i) => {
    const [cx, cy] = state.box_centers[i];
    const stepAssigned = assignedBoxToStep[i];
    let color = "#7f8c8d";
    let dim = true;
    let label = null;
    let onClick = null;

    if (stepAssigned !== undefined) {
      color = colors[stepAssigned];
      dim = false;
    } else if (inSelect && !state.lm_mode && !stepDone) {
      color = colors[state.step];
      dim = false;
      onClick = () => dispatch("select_box", { box_idx: i });
    }

    if (inVerify || inMissing) {
      const lbl = boxLabels[i];
      if (lbl) {
        color = "#1abc9c";
        dim = false;
        label = lbl;
        onClick = null;
      }
    }
    return { boxIdx: i, cx, cy, color, dim, label, onClick };
  });

  const markers = [];
  state.selections.forEach((s, i) => {
    if (s.type === "manual") {
      markers.push({ kind: "x", x: s.pos[0], y: s.pos[1], color: colors[i], label: String(i + 1) });
    }
  });
  for (const [lbl, pos] of Object.entries(state.manually_placed)) {
    markers.push({ kind: "x", x: pos[0], y: pos[1], color: "#27ae60", label: lbl });
  }
  if (inMissing) {
    for (const lbl of state.missing_labels) {
      if (state.manually_placed[lbl]) continue;
      const hint = state.missing_hint[lbl];
      if (!hint) continue;
      const isCur = lbl === state.current_missing_label;
      markers.push({
        kind: isCur ? "hint-current" : "hint",
        x: hint[0],
        y: hint[1],
        color: isCur ? "#e67e22" : "#f39c12",
        label: lbl,
      });
    }
  }

  const onImageClick = (() => {
    if (inMissing) return (x, y) => dispatch("select_image_point", { x, y });
    if (inSelect && !state.lm_mode && !stepDone)
      return (x, y) => dispatch("select_image_point", { x, y });
    return null;
  })();

  const labeledLabels = new Set(Object.values(boxLabels));
  const placedLabels = new Set(Object.keys(state.manually_placed));

  const meaStyle = (i) => {
    const lbl = state.pattern_labels[i];
    if (inMissing) {
      if (labeledLabels.has(lbl) || placedLabels.has(lbl))
        return { color: "#1abc9c", size: 5, label: lbl };
      if (lbl === state.current_missing_label)
        return { color: "#e67e22", size: 9, ring: true, label: lbl };
      if (state.missing_labels.includes(lbl))
        return { color: "#f39c12", size: 5, label: lbl };
      return { color: "#bdc3c7", size: 3, label: lbl };
    }
    const lmPos = state.lm_indices.indexOf(i);
    if (lmPos >= 0) {
      const isCurrent = lmPos === state.step && !stepDone;
      return {
        color: colors[lmPos],
        size: 9,
        ring: isCurrent,
        label: lbl,
      };
    }
    if (state.lm_mode) {
      return {
        color: "#95a5a6",
        size: 5,
        label: lbl,
        onClick: () => dispatch("change_landmark", { pattern_idx: i }),
      };
    }
    return { color: "#bdc3c7", size: 3, label: lbl };
  };

  const status = (() => {
    if (state.lm_mode)
      return "Cliquez sur une électrode du SCHÉMA MEA (à droite) pour l'utiliser comme point de référence";
    if (inSelect && !stepDone) {
      const lbl = state.pattern_labels[state.lm_indices[state.step]];
      return `Étape ${state.step + 1}/${state.n_landmarks} : trouvez l'électrode '${lbl}' sur l'image — cliquez sur son bouton, cliquez directement sur l'image, ou utilisez « Électrode introuvable ».`;
    }
    if (inSelect && stepDone) return "Tous les points de référence sont sélectionnés — cliquez sur Confirmer pour appliquer la transformation";
    if (inVerify) return "Vérifiez les libellés au-dessus de chaque boîte. Cliquez sur Confirmation finale, ou Retour.";
    if (inMissing)
      return `Placer '${state.current_missing_label}' manquante (${state.missing_step + 1}/${state.missing_labels.length}) — cliquez sur l'image, Placer automatiquement, ou Passer`;
    return "";
  })();

  return <div class="flex flex-col h-full">
      <StatusBar tone={state.lm_mode ? "warn" : inMissing ? "warn" : "info"}>
        {status}
      </StatusBar>

      <div class="flex-1 overflow-auto p-6 grid grid-cols-1 xl:grid-cols-2 gap-6 justify-items-center items-start">
        <div class="flex flex-col items-center gap-3">
          <RotateBar rot={rot} setRot={setRot}>
            <SecondaryButton
              onClick={() => dispatch("undo")}
              disabled={!inSelect || state.selections.length === 0}
            >
              Annuler
            </SecondaryButton>
            <SecondaryButton
              onClick={() => dispatch("back")}
              disabled={!inVerify}
            >
              Retour
            </SecondaryButton>
          </RotateBar>
          <ImageStage
            imageSize={state.image_size}
            rot={rot}
            boxes={boxes}
            markers={markers}
            onImageClick={onImageClick}
          />
        </div>
        <div class="flex flex-col items-center gap-3">
          <div class="flex items-center gap-2">
            <span class="text-sm text-slate-700">MEA: {state.mea_family}</span>
            <SecondaryButton
              onClick={() => dispatch("toggle_lm_mode")}
              active={state.lm_mode}
              disabled={!inSelect || stepDone}
            >
              Électrode introuvable
            </SecondaryButton>
          </div>
          <MeaStage
            patternPoints={state.pattern_points}
            patternLabels={state.pattern_labels}
            pointStyleFor={meaStyle}
          />
          
        </div>
      </div>

      <div class="flex justify-center gap-4 pb-6">
        {inMissing
          ? <>
              <PrimaryButton color="amber" onClick={() => dispatch("autoplace")}>
                Placer automatiquement
              </PrimaryButton>
              <PrimaryButton color="slate" onClick={() => dispatch("skip")}>
                Passer (non visible)
              </PrimaryButton>
          </> : <>
              <PrimaryButton
                color="emerald"
                onClick={() => dispatch("confirm")}
                disabled={(inSelect && !stepDone) || inMissing}
              >
                {inVerify ? "Confirmation finale" : "Confirmer"}
              </PrimaryButton>
              <PrimaryButton color="red" onClick={() => dispatch("cancel")}>
                Abandonner
              </PrimaryButton>
            </>}
      </div>
    </div>
  ;
}

// ── MEA family selection view ────────────────────────────────────────────────

function MeaSelectView({ state, dispatch }) {
  const conf = (state.confidence * 100).toFixed(2);

  return <div class="flex flex-col h-full">
      <StatusBar>Sélection du type de MEA</StatusBar>

      <div class="flex flex-col items-center gap-6 p-8 overflow-auto flex-1">
        <img
          src="/api/image"
          alt="image d'entrée"
          class="rounded-md shadow-lg border border-slate-200"
          style={{ maxWidth: "320px", maxHeight: "240px", objectFit: "contain" }}
        />
        <div class="text-sm text-slate-500">
          Confiance de la détection automatique : <span class="font-semibold text-slate-800">{conf}%</span>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
          {state.classes.map((cls) => {
            const isDetected = cls === state.detected_class;
            return <button
                key={cls}
                type="button"
                onClick={() => dispatch("select_class", { mea_class: cls })}
                class={[
                  "group relative p-4 rounded-xl border-2 transition cursor-pointer",
                  "hover:scale-[1.03] hover:shadow-xl",
                  isDetected
                    ? "border-purple-500 bg-purple-50 shadow-purple-200 shadow-lg"
                    : "border-slate-200 bg-white hover:border-slate-400",
                ].join(" ")}
              >
                <img
                  src={`/api/asset/${encodeURIComponent(cls)}`}
                  alt={cls}
                  class="w-64 h-64 bg-white rounded-md mx-auto"
                  style={{ maxWidth: "none" }}
                />
                <div class="mt-3 font-bold text-slate-900">{cls}</div>
                <div class="mt-1 text-xs italic h-4 text-purple-700">
                  {isDetected ? `Détection auto — ${conf}%` : ""}
                </div>
              </button>
            ;
          })}
        </div>
      </div>
    </div>
  ;
}

// ── Segmentation display view ────────────────────────────────────────────────

function SegmentationDisplayView({ state, rot, setRot, dispatch }) {
  const boxes = state.detected_boxes.map((b, i) => ({
    boxIdx: i,
    cx: (b[0] + b[2]) / 2,
    cy: (b[1] + b[3]) / 2,
    color: "#1abc9c",
    label: null,
    dim: false,
  }));
  const polys = (state.bars_poly ?? []).map((points) => ({
    points,
    fill: "rgba(26, 188, 156, 0.18)",
    stroke: "#1abc9c",
    strokeWidth: 2,
  }));

  const n = state.detected_boxes.length;
  return <div class="flex flex-col items-center h-full gap-2">
      <StatusBar>
        Détourage des électrodes — {n} électrode{n > 1 ? "s" : ""} détectée{n > 1 ? "s" : ""}.
      </StatusBar>

      <RotateBar rot={rot} setRot={setRot} />
      <ImageStage
        imageSize={state.image_size}
        rot={rot}
        boxes={boxes}
        markers={[]}
        polys={polys}
      />

      <PrimaryButton color="emerald" onClick={() => dispatch("confirm")}>
        Continuer
      </PrimaryButton>
    </div>
  ;
}

// ── Waiting view (background work running on the server) ────────────────────

function WaitingView({ state }) {
  return <div class="flex h-full items-center justify-center">
      <div class="bg-white rounded-xl px-10 py-8 text-center shadow-lg max-w-md">
        <div class="text-xl font-semibold text-slate-900">{state.message}</div>
        <div class="mt-4 text-slate-700 flex items-center justify-center gap-2">
          <Spinner/> Traitement en cours…
        </div>
      </div>
    </div>
  ;
}

// ── Cells display view ──────────────────────────────────────────────────────

function CellsDisplayView({ state, dispatch }) {
  const items = state.items ?? [];
  return <div class="flex flex-col h-full">
      <StatusBar>
        Segmentation cellulaire — {items.length} image{items.length > 1 ? "s" : ""} générée{items.length > 1 ? "s" : ""}.
      </StatusBar>

      <div class="flex-1 overflow-auto p-6">
        {items.length === 0
          ? <div class="text-slate-500 text-center py-16">
              Aucune image générée par cell_segmenter.
            </div>
          : <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 max-w-6xl mx-auto">
              {items.map((it) =>
                <figure key={it.key} class="bg-white rounded-xl shadow p-3 flex flex-col gap-2">
                  <img
                    src={`/api/asset/${encodeURIComponent(it.key)}`}
                    alt={it.label}
                    class="w-full h-auto rounded-md bg-black"
                    loading="lazy"
                  />
                  <figcaption class="text-xs text-slate-700 font-mono break-all">
                    {it.label}
                  </figcaption>
                </figure>
              )}
            </div>}
      </div>

      <div class="flex justify-center gap-4 pb-6">
        <PrimaryButton color="emerald" onClick={() => dispatch("confirm")}>
          Continuer
        </PrimaryButton>
      </div>
    </div>
  ;
}

// ── Download view ───────────────────────────────────────────────────────────

function DownloadView({ state, onRestart }) {
  const summary = state.summary ?? {};
  const vizItems = state.viz_items ?? [];
  const unconnected = state.unconnected_viz ?? null;
  const [idx, setIdx] = useState(0);
  const total = vizItems.length;
  const cur = total > 0 ? vizItems[idx] : null;

  const prev = () => setIdx((i) => (i - 1 + total) % total);
  const next = () => setIdx((i) => (i + 1) % total);

  return <div class="flex flex-col h-full">
      <StatusBar>Workflow terminé</StatusBar>

      <div class="flex-1 overflow-auto p-6">
        <div class="flex flex-col gap-8 max-w-6xl mx-auto">

          <div class="flex flex-col xl:flex-row gap-6 items-start justify-center">

            {cur
              ? <div class="xl:flex-1 min-w-0 flex flex-col gap-3">
                  <div class="flex items-center justify-center gap-3">
                    <button
                      type="button"
                      onClick={prev}
                      disabled={total <= 1}
                      class="px-3 py-2 rounded-md bg-slate-100 hover:bg-slate-200 text-slate-900 font-semibold text-lg disabled:opacity-40"
                    >
                      ←
                    </button>
                    <select
                      value={idx}
                      onChange={(e) => setIdx(Number(e.target.value))}
                      class="px-3 py-2 rounded-md bg-white border border-slate-300 text-slate-900 font-semibold min-w-[16rem]"
                    >
                      {vizItems.map((item, i) =>
                        <option key={item.url} value={i}>
                          {item.label}
                        </option>
                      )}
                    </select>
                    <button
                      type="button"
                      onClick={next}
                      disabled={total <= 1}
                      class="px-3 py-2 rounded-md bg-slate-100 hover:bg-slate-200 text-slate-900 font-semibold text-lg disabled:opacity-40"
                    >
                      →
                    </button>
                    <span class="text-slate-500 text-sm font-mono">
                      {idx + 1} / {total}
                    </span>
                  </div>
                  <img
                    src={cur.url}
                    alt={cur.label}
                    class="w-full rounded-xl shadow-lg bg-black"
                  />
                </div>
              : null}

            <div class="bg-white rounded-2xl shadow-xl p-8 w-full xl:w-80 shrink-0 flex flex-col gap-6">
              <div class="text-center">
                <div class="text-2xl font-bold text-orange-600">✓ Analyse terminée</div>
                <div class="mt-1 text-sm text-slate-500">
                  Le rapport complet est disponible au téléchargement.
                </div>
              </div>

              {Object.keys(summary).length > 0
                ? <dl class="grid grid-cols-2 gap-x-4 gap-y-1 text-sm">
                    {Object.entries(summary).map(([k, v]) =>
                      <Fragment key={k}>
                        <dt class="text-slate-500 font-mono">{k}</dt>
                        <dd class="text-slate-900 font-mono break-all">{String(v)}</dd>
                      </Fragment>
                    )}
                  </dl>
                : null}

              <div class="flex flex-col gap-3">
                <a
                  href={state.zip_url}
                  download
                  class="px-5 py-2 rounded-md font-semibold text-white shadow bg-orange-500 hover:bg-orange-400 text-center"
                >
                  Télécharger le résultat (zip)
                </a>
                <RestartButton onRestart={onRestart} variant="inline" />
              </div>
            </div>

          </div>

          {unconnected
            ? <div class="flex flex-col gap-3 border-t border-slate-200 pt-6">
                <div class="text-center text-lg font-bold text-red-600">
                  {unconnected.label}
                </div>
                <img
                  src={unconnected.url}
                  alt={unconnected.label}
                  class="w-full max-w-3xl mx-auto rounded-xl shadow-lg bg-black"
                />
              </div>
            : null}

        </div>
      </div>
    </div>
  ;
}

// ── Upload view ──────────────────────────────────────────────────────────────

function UploadView({ state, applyState }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  const send = useCallback(async (file) => {
    if (!file || busy) return;
    setBusy(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const r = await fetch("/api/upload", { method: "POST", body: fd });
      if (!r.ok) throw new Error(`upload: ${r.status} ${await r.text()}`);
      applyState(await r.json());
    } catch (e) {
      setError(String(e));
      setBusy(false);
    }
  }, [busy, applyState]);

  const onDrop = (e) => {
    e.preventDefault();
    const f = e.dataTransfer?.files?.[0];
    if (f) send(f);
  };

  return <div class="flex h-full items-center justify-center p-8">
      <div class="w-full max-w-xl bg-white rounded-2xl shadow-xl p-8 flex flex-col gap-6">
        <div class="text-center">
          <div class="text-2xl font-bold text-slate-900">Importer l'image du microscope</div>
          <div class="mt-1 text-sm text-slate-500">
            Choisissez une image JPG/PNG du MEA.
          </div>
        </div>

        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={onDrop}
          class={[
            "border-2 border-dashed rounded-xl py-12 px-6 text-center transition cursor-pointer",
            busy
              ? "border-orange-500 bg-orange-50"
              : "border-slate-300 hover:border-slate-400 bg-slate-50",
          ].join(" ")}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept={state.accept || "image/*"}
            class="hidden"
            onChange={(e) => send(e.target.files?.[0])}
          />
          {busy
            ? <div class="flex flex-col items-center gap-3 text-slate-800">
                <Spinner/>
                <span>Envoi en cours…</span>
              </div>
            : <div class="text-slate-700">
                <div class="text-lg font-semibold">Cliquez pour choisir un fichier</div>
                <div class="mt-1 text-sm text-slate-500">…ou déposez le fichier ici.</div>
              </div>}
        </div>

        {error
          ? <div class="text-sm text-red-700 bg-red-50 border border-red-200 rounded-md px-3 py-2">{error}</div>
          : null}
      </div>
    </div>
  ;
}

// ── Between/after steps ──────────────────────────────────────────────────────

function Spinner() {
  return <svg class="animate-spin h-5 w-5 text-slate-700" viewBox="0 0 24 24" fill="none">
      <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"/>
      <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"/>
    </svg>;
}

function StepDoneWaiting() {
  return <div class="flex h-full items-center justify-center">
      <div class="bg-white rounded-xl px-8 py-6 text-center shadow-lg">
        <div class="text-2xl font-bold text-orange-600">✓ Étape terminée</div>
        <div class="mt-2 text-slate-700 flex items-center justify-center gap-2">
          <Spinner/> En attente de l'étape suivante…
        </div>
      </div>
    </div>
  ;
}

function IdleWaiting() {
  return <div class="flex h-full items-center justify-center">
      <div class="bg-white rounded-xl px-8 py-6 text-center shadow-lg">
        <div class="text-xl font-semibold text-slate-800">Interface Cellvision</div>
        <div class="mt-2 text-slate-700 flex items-center justify-center gap-2">
          <Spinner/> En attente du démarrage du workflow…
        </div>
      </div>
    </div>
  ;
}

// ──────────────────────────────────────────────────────────────────────────────
// 6. App + bootstrap
// ──────────────────────────────────────────────────────────────────────────────

function App() {
  const [state, setState] = useState(null);
  const [gen, setGen] = useState(-1);
  const [error, setError] = useState(null);
  const [rot, setRot] = useState(0);
  const inflight = useRef(false);

  // Apply a fresh state snapshot, tracking generation in React state
  // so dependent effects (rotation reset, polling) can react.
  const applyState = useCallback((next) => {
    setState(next);
    if (typeof next?.generation === "number") setGen(next.generation);
  }, []);

  useEffect(() => {
    apiFetchState().then(applyState).catch((e) => setError(String(e)));
  }, [applyState]);

  useEffect(() => {
    if (state?.title) document.title = state.title;
  }, [state?.title]);

  // Reset rotation when a new step begins.
  useEffect(() => {
    if (gen >= 0) setRot(0);
  }, [gen]);

  const dispatch = useCallback(async (name, payload) => {
    if (inflight.current) return;
    inflight.current = true;
    try {
      applyState(await apiDispatch(name, payload || {}));
    } catch (e) {
      setError(String(e));
    } finally {
      inflight.current = false;
    }
  }, [applyState]);

  // User-triggered restart: aborts the active step server-side, the
  // orchestrator re-runs the workflow, and we drop into a passive
  // local state so the polling effect picks up the new generation.
  const requestRestart = useCallback(async () => {
    if (!window.confirm("Recommencer l'analyse depuis le début ?")) return;
    try {
      await apiRestart();
      applyState({
        kind: "idle",
        phase: "idle",
        title: "Recommencer…",
        generation: gen,
      });
    } catch (e) {
      setError(String(e));
    }
  }, [applyState, gen]);

  // Poll between steps so the UI follows along when the workflow swaps
  // the active logic. We poll for any state that doesn't advance via a
  // user dispatch — done/idle (between steps), waiting (background work
  // running on the server), and upload (file picker hands state back via
  // applyState, but a stale polled answer is harmless).
  useEffect(() => {
    if (!state) return;
    const passive =
      state.phase === "done" ||
      state.phase === "idle" ||
      state.kind === "waiting";
    if (!passive) return;

    let stopped = false;
    let failures = 0;
    let timer = null;

    const tick = async () => {
      if (stopped) return;
      try {
        const next = await apiFetchState();
        failures = 0;
        if (typeof next.generation === "number" && next.generation > gen) {
          applyState(next);
        }
      } catch {
        failures += 1;
        if (failures > 6) {
          setError("Connexion perdue — le workflow s'est probablement terminé.");
          return;
        }
      }
      if (!stopped) timer = setTimeout(tick, 800);
    };
    timer = setTimeout(tick, 800);
    return () => {
      stopped = true;
      if (timer) clearTimeout(timer);
    };
  }, [state?.phase, gen, applyState]);

  if (error) {
    return <div class="p-8 text-red-700">
      <div class="text-lg font-bold">Erreur</div>
      <div class="mt-2 font-mono text-sm">{error}</div>
    </div>;
  }

  if (!state) {
    return <div class="p-8 text-slate-500">Chargement…</div>;
  }

  // Header restart button: visible whenever the workflow is engaged
  // (i.e. anything past the very first idle handshake). On the upload
  // page it's redundant — the user is already at the start — so hide
  // it there.
  const showHeaderRestart =
    state.kind !== "idle" &&
    state.kind !== "upload" &&
    state.phase !== "idle";

  let view;
  if (state.kind === "idle" || state.phase === "idle") {
    view = <IdleWaiting />;
  } else if (state.phase === "done") {
    view = <StepDoneWaiting/>;
  } else if (state.kind === "verification") {
    view = <VerificationView
      state={state}
      rot={rot}
      setRot={setRot}
      dispatch={dispatch}
    />;
  } else if (state.kind === "selection") {
    view = <SelectionView
      state={state}
      rot={rot}
      setRot={setRot}
      dispatch={dispatch}
    />;
  } else if (state.kind === "mea_select") {
    view = <MeaSelectView state={state} dispatch={dispatch} />;
  } else if (state.kind === "upload") {
    view = <UploadView state={state} applyState={applyState} />;
  } else if (state.kind === "display_segmentation") {
    view = <SegmentationDisplayView
      state={state}
      rot={rot}
      setRot={setRot}
      dispatch={dispatch}
    />;
  } else if (state.kind === "display_cells") {
    view = <CellsDisplayView state={state} dispatch={dispatch} />;
  } else if (state.kind === "waiting") {
    view = <WaitingView state={state} />;
  } else if (state.kind === "download") {
    view = <DownloadView state={state} onRestart={requestRestart} />;
  } else {
    view = <div class="p-8">Type d'application inconnu : {state.kind}</div>;
  }

  return <>
    {showHeaderRestart ? <RestartButton onRestart={requestRestart} variant="header" /> : null}
    {view}
  </>;
}

export default App;