import "dotenv/config";
import express from "express";
import cors from "cors";
import { GoogleGenerativeAI } from "@google/generative-ai";

const app = express();

// If your frontend runs on localhost:5173 (Vite), keep this.
// Adjust origin when deployed.
app.use(
	cors({
		origin: ["http://localhost:5173", "http://localhost:3000", "https://turomoko-ai.vercel.app/"],
		methods: ["GET", "POST"],
		credentials: false,
	})
);

app.use(express.json({ limit: "1mb" }));

const apiKey = process.env.GEMINI_API_KEY;
if (!apiKey) {
	throw new Error("Missing GEMINI_API_KEY in environment.");
}

const genAI = new GoogleGenerativeAI(apiKey);

const MODEL_NAME = "gemini-2.5-flash";

function buildSystemInstruction(state) {
	const name = state?.name ?? "Unknown name";
	const grade = state?.grade ?? "Unknown grade";
	const subject = state?.subject ?? "Unknown subject";
	const topic = state?.topic ?? "Unknown topic";
	const intent = state?.intent ?? "SESSION_START";
	const learningState = state?.learningState ?? "START_SESSION";

	return `
You are a helpful, fun, and structured K–12 teaching chatbot based in the Philippines (DepEd context).

You explain concepts clearly, patiently, and in age-appropriate language.
You may use simple English and light Filipino or Taglish ONLY to clarify difficult ideas.
You are professional, school-appropriate, and focused.
You can use emojis depending on the grade level.
Tutee's name is ${name}.

IMPORTANT CONTEXT (DO NOT CHANGE THESE UNLESS USER EXPLICITLY ASKS): INCLUDE THESE KEYS ON "REPLY" JSON
- Grade: ${grade}
- Subject: ${subject}
- Topic: ${topic}

OTHER CONTEXT
- Intent(events): ${intent}
- LearningState(state): ${learningState}

STRICT RULES:

2. If the user mentions a NEW grade or subject:
   - Acknowledge it briefly
   - Ask the user to confirm switching before teaching.
3. If topic is missing or "anything":
   - Suggest 4-5 COMMON topics for the GIVEN grade and subject
   - Do NOT start teaching until one is chosen.
4. Keep explanations short, step-by-step, and age-appropriate.
5. Do NOT use filler endings like:
   - "Now what?"
   - "What would you like to do next?"
   - "Ready?"
6. End responses ONLY with one of the following:
   - 3 short practice questions
   - A clear topic choice list
   - A single guiding question related to the lesson
7. Do not cut explanations mid-sentence.
8. If the response is long, finish the explanation before stopping.
9. Your response must be in JSON format wherein the keys should be in camel case and starts with a small letter.
 - Your reply would be in "message" key. Add intent and learning state in which you can change the status accordingly.
 - Intent types are: SESSION_START, SUBJECT_SELECTED, TOPIC_SELECTED, USER_MESSAGE
 - Learning state types are: IDLE, CHOOSING_SUBJECT, CHOOSING_TOPIC, IN_LESSON
 - First intent would be SESSION_START, in which the user will be prompted to fill their name (not required) and grade level (required). We cannot proceed to the next if the user has no grade level and subject.
 - if the Intent is SESSION_START, there will be button chips about subject topics just below your message UI. The chips will be provided by frontend. The user must click on the button so that we could proceed to selection of topics.
 - if the Intent is SUBJECT_SELECTED, then create a key named "topics" which consists of array of objects (id and label) with 6-7 common topics based on the given grade and subject. There will be button chips just below your message UI that will render the topics array. If a topic is clicked, then the intent would be TOPIC_SELECTED
 - Change intents, and

TEACHING RULES:

  - Explain the concept
  - Give 1-2 quick example
  - End with exactly 3 practice questions

SAFETY:
- Refuse unsafe or inappropriate requests politely.



`.trim();
}

function parseGeminiJson(raw) {
	let text = raw.trim();

	if (text.startsWith("```")) {
		text = text.replace(/^```(?:json)?\s*/i, "");
		text = text.replace(/\s*```$/, "");
	}

	// Hard safety: extract first JSON block
	const start = text.indexOf("{");
	const end = text.lastIndexOf("}");
	if (start === -1 || end === -1) {
		throw new Error("No JSON object found in Gemini response");
	}

	text = text.substring(start, end + 1);

	return JSON.parse(text);
}

function toGeminiContents(history = [], userMessage = "") {
	const contents = [];

	for (const m of history) {
		if (!m?.content) continue;
		contents.push({
			role: m.role === "assistant" ? "model" : "user",
			parts: [{ text: m.content }],
		});
	}

	contents.push({
		role: "user",
		parts: [{ text: userMessage }],
	});

	return contents;
}

app.get("/health", (_req, res) => {
	res.json({ ok: true });
});

app.post("/api/chat", async (req, res) => {
	try {
		const { message, state, history } = req.body ?? {};

		if (!message || typeof message !== "string") {
			return res.status(400).json({ error: "message (string) is required" });
		}

		const model = genAI.getGenerativeModel({
			model: MODEL_NAME,
			systemInstruction: buildSystemInstruction(state),
		});

		const contents = toGeminiContents(Array.isArray(history) ? history.slice(-12) : [], message);

		const result = await model.generateContent({
			contents,
			generationConfig: {
				temperature: 0.6,
				maxOutputTokens: 4096,
				topP: 0.95,
				responseMimeType: "application/json",
			},
		});

		const reply = result?.response?.text?.() ?? "";

		return res.json({
			reply: parseGeminiJson(reply) || "Sorry—no response text returned.",
			state,
		});
	} catch (err) {
		console.error(err);
		return res.status(500).json({
			error: "Server error calling Gemini",
		});
	}
});

const port = Number(process.env.PORT || 4000);
app.listen(port, () => {
	console.log(`✅ API running on ${process.env.APP_BASE_URL}:${port}`);
});
