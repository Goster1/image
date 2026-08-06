/**
 * Camera source management for the monitor page: USB webcam via getUserMedia
 * (with device picker) or a local video file for deterministic testing.
 */

export interface CameraSource {
  video: HTMLVideoElement;
  kind: 'camera' | 'file';
  stop(): void;
}

export async function listCameras(): Promise<MediaDeviceInfo[]> {
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === 'videoinput');
}

export async function openCamera(
  video: HTMLVideoElement,
  deviceId: string | null,
): Promise<CameraSource> {
  const constraints: MediaStreamConstraints = {
    audio: false,
    video: deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1280 }, height: { ideal: 720 } }
      : { width: { ideal: 1280 }, height: { ideal: 720 } },
  };
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    if (deviceId) {
      // The stored device may be unplugged — fall back to any camera.
      stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: true });
    } else {
      throw err;
    }
  }
  video.srcObject = stream;
  await video.play();
  return {
    video,
    kind: 'camera',
    stop() {
      for (const track of stream.getTracks()) track.stop();
      video.srcObject = null;
    },
  };
}

export async function openVideoFile(video: HTMLVideoElement, file: File): Promise<CameraSource> {
  const url = URL.createObjectURL(file);
  video.srcObject = null;
  video.src = url;
  video.loop = true;
  video.muted = true;
  await video.play();
  return {
    video,
    kind: 'file',
    stop() {
      video.pause();
      video.removeAttribute('src');
      URL.revokeObjectURL(url);
    },
  };
}
