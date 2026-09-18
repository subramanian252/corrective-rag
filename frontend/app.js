const form = document.querySelector("#query-form");
const question = document.querySelector("#question");
const status = document.querySelector("#status");
const answerPanel = document.querySelector("#answer-panel");
const answer = document.querySelector("#answer");
const button = form.querySelector("button");

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const value = question.value.trim();
  if (!value) return;

  button.disabled = true;
  status.textContent = "The graph is working…";
  answerPanel.hidden = true;

  try {
    const response = await fetch("/query", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: value }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The request failed.");
    answer.textContent = data.answer;
    answerPanel.hidden = false;
    status.textContent = "Answer complete.";
  } catch (error) {
    answer.textContent = error.message;
    answerPanel.hidden = false;
    status.textContent = "Could not complete the request.";
  } finally {
    button.disabled = false;
  }
});
