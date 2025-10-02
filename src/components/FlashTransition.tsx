import { useEffect } from "react";

interface FlashTransitionProps {
  isActive: boolean;
  onComplete: () => void;
}

const FlashTransition = ({ isActive, onComplete }: FlashTransitionProps) => {
  useEffect(() => {
    if (isActive) {
      const timer = setTimeout(() => {
        onComplete();
      }, 800);
      return () => clearTimeout(timer);
    }
  }, [isActive, onComplete]);

  if (!isActive) return null;

  return (
    <div className="fixed inset-0 z-50 bg-success flash-success pointer-events-none" />
  );
};

export default FlashTransition;
