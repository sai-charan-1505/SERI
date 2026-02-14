const recordBtn = document.getElementById("recordBtn")
const stopBtn = document.getElementById("stopBtn")
const fileInput = document.getElementById("fileInput")

const gateValue = document.getElementById("gateValue")
const emotionValue = document.getElementById("emotionValue")
const confidenceValue = document.getElementById("confidenceValue")
const preview = document.getElementById("preview")

let mediaRecorder
let chunks = []

recordBtn.addEventListener("click", async () => {

    const stream = await navigator.mediaDevices.getUserMedia({ audio:true })

    chunks = []
    mediaRecorder = new MediaRecorder(stream)

    mediaRecorder.ondataavailable = e => chunks.push(e.data)

    mediaRecorder.onstop = async () => {

        const blob = new Blob(chunks, { type:"audio/webm" })
        const wavBlob = await convertTo16kMonoWav(blob)

        preview.src = URL.createObjectURL(wavBlob)

        setLoading()

        const result = await sendAudioToAPI(wavBlob)
        updateResult(result)
    }

    mediaRecorder.start()
    recordBtn.disabled = true
    stopBtn.disabled = false
})

stopBtn.addEventListener("click", () => {
    mediaRecorder.stop()
    recordBtn.disabled = false
    stopBtn.disabled = true
})

fileInput.addEventListener("change", async e => {

    const file = e.target.files[0]
    if(!file) return

    preview.src = URL.createObjectURL(file)

    const wavBlob = await convertTo16kMonoWav(file)

    setLoading()

    const result = await sendAudioToAPI(wavBlob)
    updateResult(result)
})

function setLoading(){
    gateValue.textContent = "..."
    emotionValue.textContent = "..."
    confidenceValue.textContent = "..."
}

function updateResult(result){
    gateValue.textContent = result.gate ?? "-"

    if(result.emotion){
        emotionValue.textContent = result.emotion
        confidenceValue.textContent = result.confidence.toFixed(3)
    }else{
        emotionValue.textContent = "-"
        confidenceValue.textContent = "-"
    }
}

async function convertTo16kMonoWav(blob){

    const audioContext = new AudioContext()
    const arrayBuffer = await blob.arrayBuffer()
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer)

    const offlineContext = new OfflineAudioContext(
        1,
        audioBuffer.duration * 16000,
        16000
    )

    const source = offlineContext.createBufferSource()
    source.buffer = audioBuffer
    source.connect(offlineContext.destination)
    source.start(0)

    const renderedBuffer = await offlineContext.startRendering()

    return encodeWAV(renderedBuffer.getChannelData(0))
}

function encodeWAV(samples){

    const buffer = new ArrayBuffer(44 + samples.length * 2)
    const view = new DataView(buffer)

    writeString(view, 0, "RIFF")
    view.setUint32(4, 36 + samples.length * 2, true)
    writeString(view, 8, "WAVE")
    writeString(view, 12, "fmt ")
    view.setUint32(16, 16, true)
    view.setUint16(20, 1, true)
    view.setUint16(22, 1, true)
    view.setUint32(24, 16000, true)
    view.setUint32(28, 16000 * 2, true)
    view.setUint16(32, 2, true)
    view.setUint16(34, 16, true)
    writeString(view, 36, "data")
    view.setUint32(40, samples.length * 2, true)

    let offset = 44

    for(let i=0;i<samples.length;i++){
        const s = Math.max(-1, Math.min(1, samples[i]))
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true)
        offset += 2
    }

    return new Blob([view], { type:"audio/wav" })
}

function writeString(view, offset, string){
    for(let i=0;i<string.length;i++){
        view.setUint8(offset+i, string.charCodeAt(i))
    }
}
