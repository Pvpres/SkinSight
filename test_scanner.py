#!/usr/bin/env python3
"""
Test script for the face scanner functionality
"""

from face_scanner import FaceScanner, scan_and_analyze
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_face_scanner():
    """Test the face scanner functionality"""
    try:
        print("🔍 Starting SkinSight Face Scanner Test")
        print("=" * 50)
        
        # Test 1: Initialize scanner
        print("1. Initializing face scanner...")
        scanner = FaceScanner()
        print("✅ Face scanner initialized successfully!")
        
        # Test 2: Scan and analyze
        print("\n2. Starting face scan (3 seconds)...")
        print("   - Position your face in the camera")
        print("   - Make sure only 1 face is visible")
        print("   - Press ESC to cancel")
        
        results = scanner.scan_face_from_camera(duration=3.0)
        
        # Test 3: Display results
        print("\n3. Analysis Results:")
        print("=" * 30)
        
        overall = results['overall_assessment']
        prediction = results['prediction']
        metadata = results['analysis_metadata']
        
        print(f"Overall Assessment: {'✅ Healthy' if overall['is_healthy'] else '⚠️  Has Condition'}")
        print(f"Healthy Probability: {overall['healthy_percentage']}%")
        print(f"Disease Probability: {overall['disease_percentage']}%")
        
        if not overall['is_healthy']:
            print(f"\nMost Likely Condition: {prediction['condition'].capitalize()}")
            print(f"Confidence: {prediction['confidence_percentage']}%")
            
            print("\nCondition Breakdown:")
            for condition, data in results['condition_analysis'].items():
                print(f"  {condition.capitalize()}: {data['percentage']}%")
        
        print(f"\nAnalysis Metadata:")
        print(f"  Frames Analyzed: {metadata['frames_analyzed']}")
        print(f"  Model Version: {metadata['model_version']}")
        print(f"  Device Used: {metadata['device_used']}")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Test error: {e}")

def test_convenience_function():
    """Test the convenience function"""
    try:
        print("\n🔧 Testing convenience function...")
        results = scan_and_analyze(duration=2.0)  # Shorter scan for testing
        print("✅ Convenience function test passed!")
        print(f"   Result: {results['prediction']['condition']} ({results['prediction']['confidence_percentage']}%)")
    except Exception as e:
        print(f"❌ Convenience function test failed: {e}")

if __name__ == "__main__":
    print("SkinSight Face Scanner Test Suite")
    print("=" * 40)
    
    # Run tests
    test_face_scanner()
    # test_convenience_function()  # Uncomment to test convenience function
    
    print("\n" + "=" * 40)
    print("Test suite completed!")

