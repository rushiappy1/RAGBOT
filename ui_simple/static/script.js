async function sendQuestion() {
    const input = document.getElementById("questionInput");
    const question = input.value.trim();
    if (!question) return;
    
    addMessage("You", question);
    input.value = "";

    try {
        const resp = await fetch("/ask", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({question})
        });
        const data = await resp.json();
        if(data.error) {
            addMessage("Error", data.details);
        } else {
            addMessage("Assistant", data.answer || "No answer");
        }
    } catch (err) {
        addMessage("Error", err.toString());
    }
}

function addMessage(sender, text) {
    const chatLog = document.getElementById("chatLog");
    const p = document.createElement("p");
    p.textContent = sender + ": " + text;
    chatLog.appendChild(p);
}
