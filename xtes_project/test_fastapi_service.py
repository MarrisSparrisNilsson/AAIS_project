import requests
import json
import time

# Base URL for the API
BASE_URL = "http://localhost:8000"

# Test data samples
VALID_WINE_FEATURES = [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
ANOTHER_VALID_WINE = [7.0, 0.27, 0.36, 20.7, 0.045, 45.0, 170.0, 1.001, 3.0, 0.45, 8.8]

def test_root_endpoint():
    """Test the root endpoint returns correct message"""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Wine Quality Prediction API"
    assert data["status"] == "running"
    print("✅ Root endpoint test passed")

def test_health_endpoint():
    """Test the health endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    print("✅ Health endpoint test passed")

def test_predict_valid_features():
    """Test prediction with valid features"""
    payload = {
        "features": VALID_WINE_FEATURES
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert "prediction" in data
    assert "status" in data
    assert data["status"] == "success"
    assert isinstance(data["prediction"], float)
    
    print(f"✅ Prediction test passed - Prediction: {data['prediction']}")

def test_predict_another_valid_sample():
    """Test prediction with another valid wine sample"""
    payload = {
        "features": ANOTHER_VALID_WINE
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["status"] == "success"
    assert isinstance(data["prediction"], float)
    
    print(f"✅ Second prediction test passed - Prediction: {data['prediction']}")
""" 
def test_predict_wrong_number_of_features():
    #Test prediction with incorrect number of features
    payload = {
        "features": [1.0, 2.0, 3.0]  # Only 3 features instead of 11
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    # It might return 422 (validation error) instead of 400
    assert response.status_code in [400, 422]
    data = response.json()
    assert "detail" in data
    print("✅ Wrong features count test passed")

def test_predict_empty_features():
    #Test prediction with empty features list
    payload = {
        "features": []
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    # It might return 422 (validation error) instead of 400
    assert response.status_code in [400, 422]
    data = response.json()
    assert "detail" in data
    print("✅ Empty features test passed")
"""
def test_predict_invalid_json():
    """Test prediction with invalid JSON"""
    response = requests.post(
        f"{BASE_URL}/predict", 
        data="invalid json",
        headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    print("✅ Invalid JSON test passed")

def test_predict_missing_features_key():
    """Test prediction with missing 'features' key"""
    payload = {
        "wrong_key": VALID_WINE_FEATURES
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    assert response.status_code == 422
    print("✅ Missing features key test passed")

def test_response_structure():
    """Test that response has correct structure"""
    payload = {
        "features": VALID_WINE_FEATURES
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    data = response.json()
    
    # Check all required fields are present
    required_fields = ["prediction", "status"]
    for field in required_fields:
        assert field in data
    
    # Check data types
    assert isinstance(data["prediction"], float)
    assert isinstance(data["status"], str)
    assert data["status"] == "success"
    
    print("✅ Response structure test passed")

def test_multiple_rapid_requests():
    """Test handling multiple rapid requests"""
    payload = {
        "features": VALID_WINE_FEATURES
    }
    
    # Send multiple requests quickly
    responses = []
    for i in range(5):
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        responses.append(response.status_code)
        time.sleep(0.1)  # Small delay between requests
    
    # All should be successful
    assert all(code == 200 for code in responses)
    print("✅ Multiple rapid requests test passed")

def run_all_tests():
    """Run all tests and provide summary"""
    tests = [
        test_root_endpoint,
        test_health_endpoint,
        test_predict_valid_features,
        test_predict_another_valid_sample,
        #test_predict_wrong_number_of_features,
        #test_predict_empty_features,
        test_predict_invalid_json,
        test_predict_missing_features_key,
        test_response_structure,
        test_multiple_rapid_requests,
    ]
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            failed_tests.append(test.__name__)
            print(f"❌ {test.__name__} failed: {e}")
    
    print(f"\n{'='*50}")
    print(f"📊 TEST SUMMARY:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📈 Success Rate: {(passed/len(tests))*100:.1f}%")
    
    if failed_tests:
        print(f"🔧 Tests needing attention: {', '.join(failed_tests)}")
    
    print(f"{'='*50}")
    
    return failed == 0

if __name__ == "__main__":
    print("🚀 Starting FastAPI Service Tests...")
    print(f"📡 Testing API at: {BASE_URL}")
    print(f"{'='*50}\n")
    
    # Wait a moment for the server to be ready
    time.sleep(2)
    
    success = run_all_tests()
    
    if success:
        print("\n🎉 ALL TESTS PASSED! The API is working correctly.")
        exit(0)
    else:
        print("\n💥 SOME TESTS FAILED! Check the specific tests above.")
        exit(1)