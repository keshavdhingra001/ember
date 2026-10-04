use std::fmt;

#[derive(Debug)]
pub enum Error {
    /// No GPU adapter matched (no driver, or running somewhere without a GPU).
    NoAdapter(String),
    /// The adapter refused our device request (usually an unsupported limit or feature).
    Device(String),
    /// Shapes don't fit the operation, or data length doesn't match the shape.
    Shape(String),
    /// Reading a buffer back from the GPU failed.
    Readback(String),
    /// Opening or reading a file failed. The message names the path.
    Io(String),
    /// A file was readable but malformed (bad safetensors header, bad config, bad vocab).
    Format(String),
    /// Input the model can't take: an unknown token id, a sequence longer than the context.
    Input(String),
}

impl Error {
    pub(crate) fn io(path: &std::path::Path, e: std::io::Error) -> Self {
        Error::Io(format!("{}: {e}", path.display()))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::NoAdapter(e) => write!(f, "no GPU adapter: {e}"),
            Error::Device(e) => write!(f, "device request failed: {e}"),
            Error::Shape(e) => write!(f, "shape error: {e}"),
            Error::Readback(e) => write!(f, "readback failed: {e}"),
            Error::Io(e) => write!(f, "io error: {e}"),
            Error::Format(e) => write!(f, "format error: {e}"),
            Error::Input(e) => write!(f, "invalid input: {e}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
