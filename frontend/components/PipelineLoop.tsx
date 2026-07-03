"use client";
import { motion, useReducedMotion } from "framer-motion";
import type { LoopStage } from "@/lib/derive";
import StageNode from "./StageNode";

export default function PipelineLoop({ stages }: { stages: LoopStage[] }) {
  const reduceMotion = useReducedMotion();
  return (
    <div className="loop glass">
      <div className="loop-track">
        {stages.map((s, i) =>
          reduceMotion ? (
            <div key={s.key} className="loop-cell">
              <StageNode stage={s} index={i} />
              {i < stages.length - 1 && (
                <div className="loop-connector">
                  <span className="loop-pulse" />
                </div>
              )}
            </div>
          ) : (
            <motion.div
              key={s.key}
              className="loop-cell"
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.08, duration: 0.4 }}
            >
              <StageNode stage={s} index={i} />
              {i < stages.length - 1 && (
                <div className="loop-connector">
                  <motion.span
                    className="loop-pulse"
                    animate={{ x: ["0%", "100%"] }}
                    transition={{ repeat: Infinity, duration: 1.8, delay: i * 0.2, ease: "easeInOut" }}
                  />
                </div>
              )}
            </motion.div>
          )
        )}
      </div>
    </div>
  );
}
