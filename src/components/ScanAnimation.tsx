import { useEffect, useMemo, useState } from "react";
import { Camera } from "lucide-react";

interface ScanAnimationProps {
  isScanning: boolean;
  apiDone?: boolean;
  onScanComplete: () => void;
}

const ScanAnimation = ({ isScanning, apiDone = false, onScanComplete }: ScanAnimationProps) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (!isScanning) {
      setProgress(0);
      return;
    }
    
    // Progress speed varies based on state
    const tickMs = 50;
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (apiDone) {
          // API is done, complete to 100%
          const next = prev + 3;
          return next >= 100 ? 100 : next;
        }
        
        // During initial phase (frame capture or waiting), progress to 60%
        if (prev < 60) {
          const next = prev + 2;
          return next > 60 ? 60 : next;
        }
        
        // After frame capture, slow progress until API responds
        const next = prev + 0.5;
        return next > 95 ? 95 : next;
      });
    }, tickMs);

    return () => clearInterval(interval);
  }, [isScanning, apiDone]);

  useEffect(() => {
    if (!isScanning) return;
    // Only complete scan if API is done and progress reached 100%
    // Don't auto-complete if there's likely an error (let parent handle it)
    if (apiDone && progress >= 100) {
      // Small delay before calling onScanComplete to ensure state is stable
      // The parent component will check for errors before proceeding
      const t = setTimeout(() => {
        onScanComplete();
      }, 500); // Slightly longer delay to allow error state to be set
      return () => clearTimeout(t);
    }
  }, [apiDone, progress, isScanning, onScanComplete]);

  const DOT_COUNT = 24;
  const dots = useMemo(() => {
    const arr: { x: number; y: number; active: boolean }[] = [];
    const radius = 132; // around the face circle
    const center = { x: 160, y: 160 };
    const greenCount = Math.round((progress / 100) * DOT_COUNT);
    for (let i = 0; i < DOT_COUNT; i++) {
      const angle = (i / DOT_COUNT) * Math.PI * 2;
      const x = center.x + radius * Math.cos(angle);
      const y = center.y + radius * Math.sin(angle);
      arr.push({ x, y, active: i < greenCount });
    }
    return arr;
  }, [progress]);

  if (!isScanning) return null;

  return (
    <div className="fixed inset-0 z-50 pointer-events-none">
      <div className="absolute inset-0 flex items-center justify-center">
        <div className="relative w-80 h-80 flex items-center justify-center">
          {/* Dots ring */}
          <svg className="absolute inset-0 w-full h-full">
            {dots.map((d, i) => (
              <circle
                key={i}
                cx={d.x}
                cy={d.y}
                r={4}
                fill={d.active ? "hsl(var(--success))" : "hsl(var(--border))"}
                opacity={d.active ? 1 : 0.7}
              />
            ))}
          </svg>

          {/* Face guide */}
          <div className="relative w-64 h-64 rounded-full border-2 border-white/70 flex items-center justify-center shadow-[0_0_30px_rgba(0,0,0,0.2)]">
            <Camera className="w-16 h-16 text-white/70" />
          </div>
        </div>
      </div>

      {/* Instructions and progress */}
      <div className="absolute bottom-24 left-1/2 -translate-x-1/2 text-center px-4">
        <p className="text-white text-lg font-medium drop-shadow">
          {progress < 60 ? "Capturing frames..." : apiDone ? "Processing complete!" : "Analyzing skin condition..."}
        </p>
        <p className="text-white/80 text-sm">
          {progress < 60 ? "Keep your face centered and steady" : 
           apiDone ? "Results ready" : 
           "AI is analyzing your skin..."} {Math.round(progress)}%
        </p>
      </div>
    </div>
  );
};

export default ScanAnimation;
