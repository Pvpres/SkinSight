import { useCallback, useEffect, useState } from "react";
import UploadZone from "@/components/UploadZone";
import ScanAnimation from "@/components/ScanAnimation";
import FlashTransition from "@/components/FlashTransition";
import ResultsView from "@/components/ResultsView";
import { analyze, analyzeBatch, fileToDataUrl, compressImage, testApiConnection } from "@/lib/api";

type AppState = "upload" | "scanning" | "flash" | "results";

const Index = () => {
  const [appState, setAppState] = useState<AppState>("upload");
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<{
    condition: string;
    confidence: number;
    description: string;
  } | null>(null);
  const [apiDone, setApiDone] = useState(false);

  // Check if error is related to face detection
  const isFaceDetectionError = useCallback((message: string): boolean => {
    const lowerMessage = message.toLowerCase();
    return lowerMessage.includes('no face') || 
           lowerMessage.includes('no valid faces') ||
           lowerMessage.includes('face detected') ||
           lowerMessage.includes('multiple faces');
  }, []);

  const handleUploadFile = async (file: File) => {
    setError(null);
    setResults(null);
    setApiDone(false);
    setSelectedImage(file);
    
    try {
      console.log('Starting file analysis...');
      const dataUrl = await compressImage(file, 800, 0.8);
      console.log('File compressed and converted to data URL, checking for face...');
      
      // IMPORTANT: Check for face detection BEFORE starting scan animation or API calls
      const { checkFramesForFace } = await import('@/lib/faceDetection');
      const hasFace = await checkFramesForFace([dataUrl]);
      
      if (!hasFace) {
        console.log('🚫 No face detected in uploaded file - NOT starting scan');
        setError("No face detected. Please upload an image with your face clearly visible and centered.");
        // Don't set appState to "scanning" - stay in upload state
        // Don't set apiDone - no animation was started
        return; // Don't send API call or start animation
      }
      
      // Face detected - proceed with scan animation and API call
      console.log('✅ Face detected - starting scan animation and API call');
      setAppState("scanning");
      const response = await analyze(dataUrl, 3);
      console.log('API response received:', response);
      
      if (!response.success || !response.results) {
        const errorMessage = response.message || "Analysis failed";
        
        // Check if this is a face detection error
        if (isFaceDetectionError(errorMessage)) {
          console.log('🚫 Face detection error detected, resetting to upload state');
          setError("Please position your face in the camera view. Make sure your face is clearly visible and centered.");
          setApiDone(true);
          setTimeout(() => {
            setAppState("upload");
            setError(null);
            setApiDone(false);
          }, 6000);
          return;
        }
        
        setError(errorMessage);
        setApiDone(true);
        return;
      }
      
      const r: any = response.results;
      const isHealthy = r?.overall_assessment?.is_healthy ?? false;
      const condition = isHealthy ? "Healthy Skin" : (r?.prediction?.condition || "Detected Condition");
      const confidence = Math.round(
        r?.prediction?.confidence_percentage ?? r?.overall_assessment?.healthy_percentage ?? 0
      );
      const description = isHealthy
        ? "Your skin shows signs of good health with balanced hydration levels."
        : "Our analysis suggests a possible skin condition. Consider tailored care and, if needed, consult a professional.";
      
      console.log('Setting results:', { condition, confidence, description });
      setResults({ condition, confidence, description });
      setApiDone(true);
    } catch (e: any) {
      console.error('Upload file error:', e);
      setError(e?.message || "Network error");
      setApiDone(true);
    }
  };

  const processScanResults = useCallback((response: any) => {
    if (!response.success || !response.results) {
      const errorMessage = response.message || "Analysis failed";
      
      // Check if this is a face detection error
      if (isFaceDetectionError(errorMessage)) {
        console.log('🚫 Face detection error detected, stopping immediately');
        // Stop immediately - don't continue processing
        setError("Please position your face in the camera view. Make sure your face is clearly visible and centered.");
        setApiDone(true);
        // Immediately reset to upload state so user can retry
        setTimeout(() => {
          setAppState("upload");
          setError(null);
          setApiDone(false);
        }, 100); // Very short delay for state update
        return; // Stop here - don't process
      }
      
      // For other errors, proceed normally
      setError(errorMessage);
      setApiDone(true);
      return;
    }
    
    const r: any = response.results;
    const isHealthy = r?.overall_assessment?.is_healthy ?? false;
    const condition = isHealthy ? "Healthy Skin" : (r?.prediction?.condition || "Detected Condition");
    const confidence = Math.round(
      r?.prediction?.confidence_percentage ?? r?.overall_assessment?.healthy_percentage ?? 0
    );
    const description = isHealthy
      ? "Your skin shows signs of good health with balanced hydration levels."
      : "Our analysis suggests a possible skin condition. Consider tailored care and, if needed, consult a professional.";
    
    console.log('Setting results:', { condition, confidence, description });
    setResults({ condition, confidence, description });
    setApiDone(true);
  }, [isFaceDetectionError]);

  const handleScanFrames = async (frames: string[]) => {
    try {
      console.log('Starting batch analysis with', frames.length, 'frames...');
      
      // Face detection is already checked in UploadZone before reaching here
      // So we can proceed directly with the API call
      const response = await analyzeBatch(frames, 3);
      console.log('Batch API response received:', response);
      
      // Double-check response (safety check in case something changed)
      if (!response.success || !response.results) {
        const errorMessage = response.message || "Analysis failed";
        
        if (isFaceDetectionError(errorMessage)) {
          console.log('🚫 No face detected in API response - stopping scan');
          setError("No face detected. Please position your face clearly in the camera view and try again.");
          setApiDone(true);
          setTimeout(() => {
            setAppState("upload");
            setError(null);
            setApiDone(false);
          }, 6000);
          return;
        }
      }
      
      // Process results
      processScanResults(response);
    } catch (e: any) {
      console.error('Scan frames error:', e);
      setError(e?.message || "Network error");
      setApiDone(true);
    }
  };

  const handleUseCachedResult = useCallback((frames: string[], response: any) => {
    console.log('⚡ handleUseCachedResult called', {
      hasFrames: frames?.length > 0,
      hasResponse: !!response,
      responseSuccess: response?.success
    });
    
    // Process and store results immediately
    // Face detection is already checked before caching, so we can trust cached results
    if (!response || !response.success || !response.results) {
      console.error('❌ Invalid cached response:', response);
      const errorMessage = response?.message || "Analysis failed";
      
      // Check if this is a face detection error
      if (isFaceDetectionError(errorMessage)) {
        console.log('🚫 Face detection error in cached result, stopping immediately');
        setError("Please position your face in the camera view. Make sure your face is clearly visible and centered.");
        setApiDone(true);
        setTimeout(() => {
          setAppState("upload");
          setError(null);
          setApiDone(false);
        }, 100);
        return; // Stop here - don't process
      }
      
      setError(errorMessage);
      setApiDone(true);
      return;
    }
    
    const r: any = response.results;
    const isHealthy = r?.overall_assessment?.is_healthy ?? false;
    const condition = isHealthy ? "Healthy Skin" : (r?.prediction?.condition || "Detected Condition");
    const confidence = Math.round(
      r?.prediction?.confidence_percentage ?? r?.overall_assessment?.healthy_percentage ?? 0
    );
    const description = isHealthy
      ? "Your skin shows signs of good health with balanced hydration levels."
      : "Our analysis suggests a possible skin condition. Consider tailored care and, if needed, consult a professional.";
    
    console.log('💾 Storing results:', { condition, confidence, description });
    // Store results immediately
    setResults({ condition, confidence, description });
    
    // Set apiDone after a short delay to allow animation to start
    // The animation will progress naturally from 0% to 95%, then complete to 100% when apiDone becomes true
    // This gives a smooth visual experience while still completing quickly
    const animationDelay = 1500; // 1.5 seconds - enough for animation to progress smoothly
    console.log(`⏱️ Setting apiDone to true in ${animationDelay}ms`);
    
    // Use a ref or closure to ensure we're setting the right state
    const timeoutId = setTimeout(() => {
      console.log('✅ Results ready - setting apiDone to true');
      setApiDone(true);
      // Double-check it was set
      setTimeout(() => {
        console.log('🔍 Verifying apiDone was set correctly');
      }, 100);
    }, animationDelay);
    
    // Fallback: ensure apiDone is set even if something goes wrong
    setTimeout(() => {
      console.log('🛡️ Fallback: ensuring apiDone is true after 3 seconds');
      setApiDone(true);
    }, 3000);
  }, []);

  const handleStartScan = () => {
    // Clear any previous errors when starting a new scan
    setError(null);
    setResults(null);
    setApiDone(false);
    setAppState("scanning");
    console.log('Scan started - waiting for frames...');
  };

  const handleScanComplete = useCallback(() => {
    // Only proceed to flash if we have valid results (no error and results exist)
    // If there's an error (like no face detected), don't proceed
    if (error) {
      console.log('⚠️ Scan complete called but error exists, not proceeding to flash');
      // Don't proceed - error handling will reset to upload state
      // The error state will prevent progression
      return;
    }
    
    // Only proceed if we have actual results
    if (results) {
      console.log('✅ Scan complete with results, proceeding to flash');
      setAppState("flash");
    } else {
      console.log('⚠️ Scan complete called but no results, not proceeding');
      // If no results and no error, something went wrong - reset
      setTimeout(() => {
        setAppState("upload");
        setApiDone(false);
      }, 1000);
    }
  }, [error, results]);

  const handleFlashComplete = () => {
    setAppState("results");
  };

  const handleNewScan = () => {
    setSelectedImage(null);
    setResults(null);
    setError(null);
    setAppState("upload");
  };

  // Test API connectivity on component mount
  useEffect(() => {
    testApiConnection();
  }, []);

  // When ScanAnimation completes, proceed to flash, then results

  return (
    <>
        {appState === "upload" && (
          <>
            <UploadZone
              onUploadFile={handleUploadFile}
              onScanFrames={handleScanFrames}
              onStartScan={handleStartScan}
              onUseCachedResult={handleUseCachedResult}
              onNoFaceDetected={() => {
                // Callback when no face is detected - don't start scan animation
                console.log('🚫 No face detected - scan animation will not start');
              }}
            />
        </>
      )}
      {appState === "scanning" && (
        <UploadZone
          readonly
          onUploadFile={handleUploadFile}
          onScanFrames={handleScanFrames}
          onStartScan={handleStartScan}
          onUseCachedResult={handleUseCachedResult}
        />
      )}
      
      <ScanAnimation
        isScanning={appState === "scanning"}
        apiDone={apiDone}
        onScanComplete={handleScanComplete}
      />
      
      <FlashTransition
        isActive={appState === "flash"}
        onComplete={handleFlashComplete}
      />
      
      {appState === "results" && (
        <div className="relative">
          {error ? (
            <div className="min-h-screen flex items-center justify-center p-8">
              <div className="max-w-lg w-full bg-card border border-border rounded-xl p-6 text-center">
                <p className="text-lg font-semibold text-destructive mb-2">{error}</p>
                <p className="text-sm text-muted-foreground mb-4">Please try again or choose a different photo.</p>
                <button
                  onClick={handleNewScan}
                  className="px-6 py-3 bg-primary text-primary-foreground rounded-full font-medium hover:opacity-90 transition-opacity"
                >
                  Try Another Photo
                </button>
              </div>
            </div>
          ) : results ? (
            <ResultsView
              condition={results.condition}
              confidence={results.confidence}
              description={results.description}
            />
          ) : (
            <div className="min-h-screen flex items-center justify-center p-8">
              <div className="text-center text-muted-foreground">Preparing results...</div>
            </div>
          )}
          <button
            onClick={handleNewScan}
            className="fixed bottom-8 left-1/2 -translate-x-1/2 px-6 py-3 bg-primary text-primary-foreground rounded-full font-medium hover:opacity-90 transition-opacity shadow-lg"
          >
            New Analysis
          </button>
        </div>
      )}
    </>
  );
};

export default Index;
