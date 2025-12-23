curl.exe -X POST "http://localhost:4000/predict" `
  -H "accept: application/json" `
  -H "Content-Type: application/json" `
  -d "@payload_test.json"