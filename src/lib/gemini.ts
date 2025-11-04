/**
 * Gemini API integration for product recommendations
 * Uses Google Generative AI for intelligent product recommendations
 * 
 * To install: npm install @google/genai
 */

// @ts-ignore - Package will be installed
import { GoogleGenAI } from "@google/genai";

// Get API key from environment (works in both Node.js and browser with Vite)
// In Vite, use VITE_GEMINI_API_KEY in .env file
// In Node.js, use GEMINI_API_KEY environment variable or .env file
const getApiKey = (): string | undefined => {
  // Try Vite environment variable first (for browser)
  if (typeof import.meta !== 'undefined' && import.meta.env?.VITE_GEMINI_API_KEY) {
    const key = import.meta.env.VITE_GEMINI_API_KEY;
    if (key && key.trim() !== '') return key.trim();
  }
  // Try Node.js environment variable
  if (typeof process !== 'undefined') {
    const key = process.env?.GEMINI_API_KEY || process.env?.VITE_GEMINI_API_KEY;
    if (key && key.trim() !== '') return key.trim();
  }
  return undefined;
};

// Lazy initialization - get AI client when needed
let ai: any = null;
let initializationAttempted = false;

function getAI(): any {
  if (ai) return ai;
  
  if (!initializationAttempted) {
    initializationAttempted = true;
    const apiKey = getApiKey();
    
    if (apiKey) {
      try {
        ai = new GoogleGenAI({ apiKey });
      } catch (error) {
        console.error('❌ Failed to initialize Gemini AI client:', error);
        ai = null;
      }
    } else {
      console.warn('⚠️ GEMINI_API_KEY not found. API calls will fail.');
      console.warn('   Set VITE_GEMINI_API_KEY in .env file for browser, or GEMINI_API_KEY for Node.js');
    }
  }
  
  return ai;
}

/**
 * Get product recommendations using Gemini API
 * This function is ready to use with your skin condition model results
 * 
 * @param condition - The detected skin condition (e.g., "acne", "dry", "eczema", "oily", "healthy")
 * @param confidence - Confidence score of the condition detection (0-100)
 * @returns Raw text response from Gemini (contains JSON array)
 */
export async function getProductRecommendationsGemini(
  condition: string, 
  confidence: number
): Promise<string> {
  const client = getAI();
  if (!client) {
    throw new Error('Gemini AI client not initialized. Please set GEMINI_API_KEY or VITE_GEMINI_API_KEY environment variable.');
  }
  
  try {
    const response = await client.models.generateContent({
      model: "gemini-2.5-flash",
      contents: `You are a skincare expert. Given a detected skin condition and confidence score, recommend 3–5 different skincare products that together form a complete regimen (no duplicates in type, e.g., not two moisturizers). For each product, include:
- Product name
- 1-sentence benefit
- Purchase link (Amazon, Sephora, or Ulta preferred)

Condition: ${condition}
Confidence: ${confidence}%

Return results in JSON format:
[
  {"product": "", "type": "", "benefit": "", "link": ""}
]
Keep it brief, diverse, and relevant to the condition.`,
    });
    return response.text;
  } catch (error) {
    console.error('❌ Error calling Gemini API:', error);
    throw error;
  }
}
