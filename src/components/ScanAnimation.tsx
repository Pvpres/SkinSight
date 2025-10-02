import { useEffect, useState } from "react";
import { Camera } from "lucide-react";

interface ScanAnimationProps {
  isScanning: boolean;
  onScanComplete: () => void;
}

const ScanAnimation = ({ isScanning, onScanComplete }: ScanAnimationProps) => {
  const [progress, setProgress] = useState(0);

  useEffect(() => {
    if (isScanning) {
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 100) {
            clearInterval(interval);
            setTimeout(() => onScanComplete(), 500);
            return 100;
          }
          return prev + 2;
        });
      }, 40);

      return () => clearInterval(interval);
    }
  }, [isScanning, onScanComplete]);

  if (!isScanning) return null;

  return (
    <div className="fixed inset-0 flex items-center justify-center bg-background z-50">
      <div className="relative">
        {/* Outer ring container */}
        <div className="relative w-80 h-80 flex items-center justify-center">
          {/* Animated scan ring */}
          <svg className="absolute inset-0 w-full h-full -rotate-90">
            <circle
              cx="160"
              cy="160"
              r="150"
              stroke="hsl(var(--border))"
              strokeWidth="3"
              fill="none"
            />
            <circle
              cx="160"
              cy="160"
              r="150"
              stroke="hsl(var(--success))"
              strokeWidth="3"
              fill="none"
              strokeDasharray={`${2 * Math.PI * 150}`}
              strokeDashoffset={`${2 * Math.PI * 150 * (1 - progress / 100)}`}
              className="transition-all duration-200 ease-linear drop-shadow-[0_0_8px_hsl(var(--success-glow))]"
              strokeLinecap="round"
            />
          </svg>

          {/* Face circle */}
          <div className="relative w-64 h-64 rounded-full border-4 border-border bg-card flex items-center justify-center overflow-hidden">
            <Camera className="w-24 h-24 text-muted-foreground" />
            
            {/* Scanning line effect */}
            <div 
              className="absolute inset-0 bg-gradient-to-b from-transparent via-success/20 to-transparent"
              style={{
                transform: `translateY(${(progress / 100) * 100 - 50}%)`,
                transition: 'transform 0.2s linear'
              }}
            />
          </div>
        </div>

        {/* Progress text */}
        <div className="absolute -bottom-16 left-1/2 -translate-x-1/2 text-center">
          <p className="text-2xl font-semibold text-foreground mb-1">{progress}%</p>
          <p className="text-sm text-muted-foreground">Analyzing skin condition...</p>
        </div>
      </div>
    </div>
  );
};

export default ScanAnimation;
