const API_BASE_URL = "http://127.0.0.1:8000"

async function sendAudioToAPI(blob){

    const formData = new FormData()
    formData.append("file", blob, "audio.webm")

    const response = await fetch(API_BASE_URL + "/predict", {
        method: "POST",
        body: formData
    })

    return await response.json()
}
 