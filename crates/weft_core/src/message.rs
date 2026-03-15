//! Message and conversation types for the OpenAI-compatible chat completions API.

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
}

/// Incoming request in OpenAI chat completions format.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct ChatCompletionRequest {
    /// Model identifier. In Weft v1, this is ignored (gateway uses its configured model).
    /// Preserved in response for client compatibility.
    pub model: String,
    pub messages: Vec<Message>,
    /// Maximum tokens in response. Optional, provider default if absent.
    #[serde(default)]
    pub max_tokens: Option<u32>,
    /// Temperature. Optional, provider default if absent.
    #[serde(default)]
    pub temperature: Option<f32>,
    /// Stream flag. Weft v1 does not support streaming.
    /// If true, the gateway returns 400 with an explanation.
    #[serde(default)]
    pub stream: Option<bool>,
}

/// Response in OpenAI chat completions format.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    /// Always "chat.completion"
    pub object: String,
    /// Unix timestamp
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: Message,
    pub finish_reason: String,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_role_serde_lowercase() {
        let msg = Message {
            role: Role::User,
            content: "hello".to_string(),
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("\"user\""));

        let msg_sys = Message {
            role: Role::System,
            content: "sys".to_string(),
        };
        let json_sys = serde_json::to_string(&msg_sys).unwrap();
        assert!(json_sys.contains("\"system\""));

        let msg_asst = Message {
            role: Role::Assistant,
            content: "asst".to_string(),
        };
        let json_asst = serde_json::to_string(&msg_asst).unwrap();
        assert!(json_asst.contains("\"assistant\""));
    }

    #[test]
    fn test_chat_completion_request_round_trip() {
        let json = r#"{
            "model": "claude-3-5-sonnet",
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "max_tokens": 1024,
            "temperature": 0.7,
            "stream": false
        }"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.model, "claude-3-5-sonnet");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.messages[0].role, Role::User);
        assert_eq!(req.messages[0].content, "Hello");
        assert_eq!(req.max_tokens, Some(1024));
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.stream, Some(false));
    }

    #[test]
    fn test_chat_completion_request_defaults() {
        let json = r#"{"model": "gpt-4", "messages": [{"role": "user", "content": "Hi"}]}"#;
        let req: ChatCompletionRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.max_tokens, None);
        assert_eq!(req.temperature, None);
        assert_eq!(req.stream, None);
    }

    #[test]
    fn test_chat_completion_response_serializes() {
        let resp = ChatCompletionResponse {
            id: "chatcmpl-abc123".to_string(),
            object: "chat.completion".to_string(),
            created: 1_700_000_000,
            model: "claude-3-5-sonnet".to_string(),
            choices: vec![Choice {
                index: 0,
                message: Message {
                    role: Role::Assistant,
                    content: "Hello!".to_string(),
                },
                finish_reason: "stop".to_string(),
            }],
            usage: Usage {
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            },
        };
        let json = serde_json::to_string(&resp).unwrap();
        assert!(json.contains("chatcmpl-abc123"));
        assert!(json.contains("chat.completion"));
        assert!(json.contains("assistant"));
    }

    #[test]
    fn test_role_deserialization() {
        #[derive(serde::Deserialize)]
        struct Test {
            role: Role,
        }
        let t: Test = serde_json::from_str(r#"{"role":"system"}"#).unwrap();
        assert_eq!(t.role, Role::System);

        let t: Test = serde_json::from_str(r#"{"role":"user"}"#).unwrap();
        assert_eq!(t.role, Role::User);

        let t: Test = serde_json::from_str(r#"{"role":"assistant"}"#).unwrap();
        assert_eq!(t.role, Role::Assistant);
    }
}
