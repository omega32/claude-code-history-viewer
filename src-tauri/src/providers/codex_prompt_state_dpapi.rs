#[cfg(windows)]
#[allow(unsafe_code)]
mod native {
    use windows_sys::Win32::Foundation::LocalFree;
    use windows_sys::Win32::Security::Cryptography::{
        CryptProtectData, CryptUnprotectData, CRYPTPROTECT_UI_FORBIDDEN, CRYPT_INTEGER_BLOB,
    };
    use zeroize::Zeroize;

    struct NativeOutput(CRYPT_INTEGER_BLOB);

    impl NativeOutput {
        fn new() -> Self {
            Self(CRYPT_INTEGER_BLOB {
                cbData: 0,
                pbData: std::ptr::null_mut(),
            })
        }

        fn copy_bounded(&self, limit: usize) -> Option<Vec<u8>> {
            let length = usize::try_from(self.0.cbData).ok()?;
            if length == 0 || length > limit || self.0.pbData.is_null() {
                return None;
            }
            let mut bytes = Vec::new();
            bytes.try_reserve_exact(length).ok()?;
            // DPAPI owns exactly cbData readable bytes until this guard releases them.
            bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(self.0.pbData, length) });
            Some(bytes)
        }
    }

    impl Drop for NativeOutput {
        fn drop(&mut self) {
            if !self.0.pbData.is_null() {
                // The output allocation remains valid on every exit, including bound failures.
                unsafe {
                    std::slice::from_raw_parts_mut(self.0.pbData, self.0.cbData as usize).zeroize();
                    LocalFree(self.0.pbData.cast());
                }
            }
        }
    }

    fn input(bytes: &[u8]) -> Option<CRYPT_INTEGER_BLOB> {
        Some(CRYPT_INTEGER_BLOB {
            cbData: u32::try_from(bytes.len()).ok()?,
            pbData: bytes.as_ptr().cast_mut(),
        })
    }

    pub(super) fn protect(bytes: &[u8], scope: &[u8], limit: usize) -> Option<Vec<u8>> {
        let data = input(bytes)?;
        let entropy = input(scope)?;
        let mut output = NativeOutput::new();
        let success = unsafe {
            CryptProtectData(
                &data,
                std::ptr::null(),
                &entropy,
                std::ptr::null(),
                std::ptr::null(),
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut output.0,
            )
        };
        if success == 0 {
            return None;
        }
        output.copy_bounded(limit)
    }

    pub(super) fn unprotect(bytes: &[u8], scope: &[u8], limit: usize) -> Option<Vec<u8>> {
        let data = input(bytes)?;
        let entropy = input(scope)?;
        let mut output = NativeOutput::new();
        let success = unsafe {
            CryptUnprotectData(
                &data,
                std::ptr::null_mut(),
                &entropy,
                std::ptr::null(),
                std::ptr::null(),
                CRYPTPROTECT_UI_FORBIDDEN,
                &mut output.0,
            )
        };
        if success == 0 {
            return None;
        }
        output.copy_bounded(limit)
    }
}

pub(super) fn protect(bytes: &[u8], scope: &[u8], limit: usize) -> Option<Vec<u8>> {
    #[cfg(windows)]
    {
        native::protect(bytes, scope, limit)
    }
    #[cfg(not(windows))]
    {
        let _ = (bytes, scope, limit);
        None
    }
}

pub(super) fn unprotect(bytes: &[u8], scope: &[u8], limit: usize) -> Option<Vec<u8>> {
    #[cfg(windows)]
    {
        native::unprotect(bytes, scope, limit)
    }
    #[cfg(not(windows))]
    {
        let _ = (bytes, scope, limit);
        None
    }
}
