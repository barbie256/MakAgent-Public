import { createAssistantMessage } from "../chatMessage";

const API_BASE_URL = (process.env.REACT_APP_API_BASE_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export async function postChatMessage(message) {
  const response = await fetch(`${API_BASE_URL}/agent/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: "1", message })
  });
  if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
  const payload = await response.json();
  if (!payload?.success || !payload?.data || typeof payload.data.reply !== "string") {
    throw new Error("The service returned an invalid response");
  }
  return createAssistantMessage(payload.data);
}
