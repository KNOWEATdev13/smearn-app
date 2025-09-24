
import { GoogleGenAI, Type } from "@google/genai";
import type { Question } from '../types';

const API_KEY = process.env.API_KEY;

if (!API_KEY) {
  throw new Error("API_KEY environment variable not set");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

const systemInstruction = `You are Smearn AI, a friendly and expert tutor for Nigerian secondary school students preparing for JAMB and WAEC exams. 
Your tone should be encouraging and helpful. Explain concepts clearly and provide step-by-step solutions when needed.
Format your responses using markdown for readability, including code blocks for calculations, bold text for key terms, and lists for steps.
Always be positive and aim to build the student's confidence. If a question is from a specific subject, frame your answer in that context.`;

export async function streamChatResponse(prompt: string) {
  try {
    const response = await ai.models.generateContentStream({
        model: "gemini-2.5-flash",
        contents: prompt,
        config: {
            systemInstruction: systemInstruction,
            temperature: 0.7,
            topP: 0.95,
        },
    });
    return response;
  } catch (error) {
    console.error("Error in streamChatResponse:", error);
    throw new Error("Failed to get response from AI. Please check your connection or API key.");
  }
}

const questionSchema = {
    type: Type.ARRAY,
    items: {
        type: Type.OBJECT,
        properties: {
            text: { type: Type.STRING, description: 'The full text of the question.' },
            options: { type: Type.ARRAY, items: { type: Type.STRING }, description: 'An array of 4 strings representing the options.' },
            answer: { type: Type.STRING, description: 'The correct option string.' },
        },
        required: ['text', 'options', 'answer']
    }
};


export async function extractQuestionsFromFile(file: { inlineData: { data: string; mimeType: string; } }): Promise<Omit<Question, 'id'>[]> {
    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: {
                parts: [
                    file,
                    { text: "Extract all the multiple-choice questions from this image. Each question should have 4 options and one correct answer. Ensure the answer matches one of the options exactly." }
                ]
            },
            config: {
                responseMimeType: "application/json",
                responseSchema: questionSchema,
            }
        });

        const jsonString = response.text.trim();
        const parsedData = JSON.parse(jsonString);
        
        if (!Array.isArray(parsedData)) {
            throw new Error("AI response is not in the expected array format.");
        }
        
        // Basic validation of the parsed data
        return parsedData.filter(q => q.text && Array.isArray(q.options) && q.options.length === 4 && q.answer);

    } catch (error) {
        console.error("Error in extractQuestionsFromFile:", error);
        throw new Error("Failed to extract questions from the file. The AI could not process the image.");
    }
}
