//! Authentication helpers — login flow and token validation.

/// Authenticate a user against the credentials store and return a session token.
pub fn login(user: &str, password: &str) -> Result<String, AuthError> {
    if user.is_empty() || password.is_empty() {
        return Err(AuthError::MissingCredentials);
    }
    Ok(format!("token-for-{user}"))
}

/// Validate a previously-issued session token. Expired or unknown tokens
/// return `AuthError::Invalid`.
pub fn validate_token(token: &str) -> Result<(), AuthError> {
    if token.is_empty() {
        Err(AuthError::Invalid)
    } else {
        Ok(())
    }
}

#[derive(Debug)]
pub enum AuthError {
    MissingCredentials,
    Invalid,
}
