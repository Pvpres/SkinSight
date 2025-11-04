/**
 * Client-side face detection using API pre-check
 * Sends a single test frame to API to check for face before sending full batch
 */

import { analyze } from './api';

/**
 * Check if a face is detected in frames by sending a test frame to the API first
 * This prevents unnecessary API calls if no face is present
 * 
 * @param frames - Array of frame data URLs
 * @returns true if face is detected, false otherwise
 */
export async function checkFramesForFace(frames: string[]): Promise<boolean> {
  if (!frames || frames.length === 0) {
    console.warn('⚠️ No frames provided for face detection check');
    return false;
  }

  console.log(`🔍 Pre-checking face detection with first frame before sending batch...`);

  try {
    // Send first frame to API to check for face detection
    // This is a quick check before sending the full batch
    const testFrame = frames[0];
    const response = await analyze(testFrame, 1); // Quick check with 1 second duration
    
    console.log('📋 Face detection pre-check response:', response);

    // Check if face was detected
    if (!response.success || !response.results) {
      const errorMessage = response.message || '';
      const isNoFaceError = 
        errorMessage.toLowerCase().includes('no face') ||
        errorMessage.toLowerCase().includes('no valid faces') ||
        errorMessage.toLowerCase().includes('face detected') ||
        errorMessage.toLowerCase().includes('multiple faces');
      
      if (isNoFaceError) {
        console.log('❌ No face detected in pre-check - blocking batch API call');
        return false;
      }
    }

    // If we got results, face was detected
    if (response.success && response.results) {
      console.log('✅ Face detected in pre-check - allowing batch API call');
      return true;
    }

    // If unsure, allow API call (fallback)
    console.warn('⚠️ Uncertain face detection result, allowing API call');
    return true;
  } catch (error: any) {
    console.error('❌ Face detection pre-check failed:', error);
    // On error, allow API call (fallback to full batch check)
    console.warn('⚠️ Falling back to full batch API call');
    return true;
  }
}

