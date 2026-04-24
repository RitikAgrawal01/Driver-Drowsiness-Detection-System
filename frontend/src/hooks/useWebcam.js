import { useRef, useState, useCallback } from 'react'

export function useWebcam() {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const streamRef = useRef(null)
  const [isOn, setIsOn] = useState(false)
  const [error, setError] = useState(null)

  const start = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setIsOn(true)
      setError(null)
    } catch (e) {
      setError(e.message || 'Camera access denied')
    }
  }, [])

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach(t => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setIsOn(false)
  }, [])

  // Capture current frame as base64 JPEG
  const captureFrame = useCallback((quality = 0.7) => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || !isOn) return null
    canvas.width  = video.videoWidth  || 640
    canvas.height = video.videoHeight || 480
    const ctx = canvas.getContext('2d')
    ctx.drawImage(video, 0, 0)
    // Remove data:image/jpeg;base64, prefix
    return canvas.toDataURL('image/jpeg', quality).split(',')[1]
  }, [isOn])

  return { videoRef, canvasRef, isOn, error, start, stop, captureFrame }
}
