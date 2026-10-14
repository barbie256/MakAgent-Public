import { useCallback, useEffect, useRef, useState } from "react";
import {
  createSpeechRecognition,
  recognitionErrorMessage,
  speechRecognitionSupported
} from "../voice/speechRecognition";
import { normalizeMakerereVocabulary } from "../voice/makerereVocabulary";

function combinedTranscript(existing, transcript) {
  if (!existing) return transcript;
  const separator = /\s$/.test(existing) ? "" : " ";
  return `${existing}${separator}${transcript}`;
}

export default function useVoiceInput({ value, onTranscript, onStatus, composerRef, busy }) {
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef(null);
  const startingValueRef = useRef("");
  const transcriptRef = useRef("");
  const manualStopRef = useRef(false);
  const errorRef = useRef(false);
  const supported = speechRecognitionSupported();

  const focusComposer = useCallback(() => {
    requestAnimationFrame(() => composerRef.current?.focus());
  }, [composerRef]);

  const releaseRecognition = useCallback(() => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    recognition.onstart = null;
    recognition.onresult = null;
    recognition.onerror = null;
    recognition.onend = null;
    recognitionRef.current = null;
  }, []);

  const stopVoiceInput = useCallback(() => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    manualStopRef.current = true;
    setListening(false);
    onStatus("Voice input stopped.");
    try {
      recognition.stop();
    } catch {
      releaseRecognition();
    }
    focusComposer();
  }, [focusComposer, onStatus, releaseRecognition]);

  const cancelVoiceInput = useCallback(() => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    releaseRecognition();
    setListening(false);
    try {
      recognition.abort();
    } catch {
      // The recognition session may already have ended.
    }
  }, [releaseRecognition]);

  const startVoiceInput = useCallback(() => {
    if (busy || listening) return;
    if (!supported) {
      onStatus("Voice input is not supported in this browser. You can continue typing.");
      return;
    }

    const recognition = createSpeechRecognition();
    startingValueRef.current = value;
    transcriptRef.current = "";
    manualStopRef.current = false;
    errorRef.current = false;
    recognitionRef.current = recognition;

    recognition.onstart = () => {
      setListening(true);
      onStatus("Listening.");
    };
    recognition.onresult = (event) => {
      let transcript = "";
      for (let index = 0; index < event.results.length; index += 1) {
        transcript += event.results[index][0]?.transcript || "";
      }
      transcript = transcript.trim();
      const normalizedTranscript = normalizeMakerereVocabulary(transcript);
      transcriptRef.current = normalizedTranscript;
      onTranscript(combinedTranscript(startingValueRef.current, normalizedTranscript));
    };
    recognition.onerror = (event) => {
      errorRef.current = true;
      setListening(false);
      onStatus(
        manualStopRef.current && event?.error === "aborted"
          ? "Voice input stopped."
          : recognitionErrorMessage(event)
      );
      releaseRecognition();
      focusComposer();
    };
    recognition.onend = () => {
      setListening(false);
      if (!errorRef.current) {
        if (manualStopRef.current) onStatus("Voice input stopped.");
        else if (transcriptRef.current) onStatus("Transcription ready.");
        else onStatus("No speech detected. Try again or type your question.");
      }
      releaseRecognition();
      focusComposer();
    };

    try {
      recognition.start();
    } catch (error) {
      errorRef.current = true;
      setListening(false);
      onStatus(recognitionErrorMessage(error));
      releaseRecognition();
      focusComposer();
    }
  }, [busy, focusComposer, listening, onStatus, onTranscript, releaseRecognition, supported, value]);

  useEffect(() => () => {
    const recognition = recognitionRef.current;
    if (!recognition) return;
    releaseRecognition();
    try {
      recognition.abort();
    } catch {
      // The recognition session may already have ended.
    }
  }, [releaseRecognition]);

  return {
    supported,
    listening,
    startVoiceInput,
    stopVoiceInput,
    cancelVoiceInput
  };
}
