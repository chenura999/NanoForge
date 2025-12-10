//! AVX-512 Instruction Encoding
//!
//! Provides raw byte emission for AVX-512 instructions using EVEX prefix.
//! dynasm-rs doesn't support EVEX encoding, so we encode manually.
//!
//! EVEX Prefix Format (4 bytes):
//! Byte 0: 0x62 (EVEX identifier)
//! Byte 1: R'RXB'00mm (mm=map)  
//! Byte 2: Wvvvv1pp (W=64-bit, vvvv=src reg)
//! Byte 3: zaaa0bVV' (a=mask, b=broadcast, VV'=hi src)

#![allow(dead_code)]

/// EVEX prefix builder for AVX-512 instructions
#[derive(Debug, Clone, Copy)]
pub struct EvexPrefix {
    /// Map select (0x01 = 0F, 0x02 = 0F38, 0x03 = 0F3A)
    map: u8,
    /// W bit (1 = 64-bit operands)
    w: bool,
    /// VEX.vvvv (source register, inverted)
    vvvv: u8,
    /// pp (prefix: 00=none, 01=66, 10=F3, 11=F2)
    pp: u8,
    /// Destination/source register (zmm0-zmm15)
    reg: u8,
    /// R/M register or memory base
    rm: u8,
    /// Use memory addressing
    is_mem: bool,
    /// Memory displacement (if is_mem)
    disp: i32,
    /// Scale-Index-Base index register (if is_mem)
    index: Option<u8>,
    /// Scale (1, 2, 4, 8)
    scale: u8,
}

impl EvexPrefix {
    pub fn new() -> Self {
        Self {
            map: 0x01, // 0F map
            w: true,   // 64-bit
            vvvv: 0,
            pp: 0x01, // 66 prefix (for packed integer)
            reg: 0,
            rm: 0,
            is_mem: false,
            disp: 0,
            index: None,
            scale: 1,
        }
    }

    pub fn with_dest(mut self, reg: u8) -> Self {
        self.reg = reg & 0x0F;
        self
    }

    pub fn with_src1(mut self, reg: u8) -> Self {
        self.vvvv = reg & 0x0F;
        self
    }

    pub fn with_src2_reg(mut self, reg: u8) -> Self {
        self.rm = reg & 0x0F;
        self.is_mem = false;
        self
    }

    pub fn with_mem_base(mut self, base: u8) -> Self {
        self.rm = base & 0x0F;
        self.is_mem = true;
        self
    }

    pub fn with_index(mut self, index: u8, scale: u8) -> Self {
        self.index = Some(index & 0x0F);
        self.scale = scale;
        self
    }

    pub fn with_disp(mut self, disp: i32) -> Self {
        self.disp = disp;
        self
    }

    pub fn with_map(mut self, map: u8) -> Self {
        self.map = map;
        self
    }

    /// Encode the EVEX prefix (4 bytes)
    fn encode_prefix(&self) -> [u8; 4] {
        // Byte 0: EVEX identifier
        let byte0 = 0x62u8;

        // Byte 1: R'RXB'00mm
        // R = NOT(reg[3]), R' = NOT(reg[4]) for zmm16-31, X = NOT(index[3]), B = NOT(rm[3])
        let r_bit = if self.reg & 0x08 != 0 { 0 } else { 0x80 };
        let x_bit = match self.index {
            Some(idx) if idx & 0x08 != 0 => 0,
            _ => 0x40,
        };
        let b_bit = if self.rm & 0x08 != 0 { 0 } else { 0x20 };
        let r_prime = 0x10; // R' = 1 for zmm0-15
        let byte1 = r_bit | r_prime | x_bit | b_bit | self.map;

        // Byte 2: Wvvvv1pp
        let w_bit = if self.w { 0x80 } else { 0 };
        let vvvv_inv = (!self.vvvv & 0x0F) << 3;
        let byte2 = w_bit | vvvv_inv | 0x04 | self.pp; // bit 2 is always 1

        // Byte 3: zaaa0bVV'
        // z=0 (no zeroing), aaa=000 (no mask), b=0 (no broadcast), VV'=11 (vvvv[4]=0)
        let byte3 = 0x00 | 0x08; // VV' bits set for zmm0-15

        [byte0, byte1, byte2, byte3]
    }

    /// Encode ModR/M byte
    fn encode_modrm(&self) -> u8 {
        let reg_field = (self.reg & 0x07) << 3;
        let rm_field = self.rm & 0x07;

        if self.is_mem {
            if self.index.is_some() {
                // SIB follows
                0x04 | reg_field // mod=00, rm=100 (SIB)
            } else if self.disp == 0 {
                0x00 | reg_field | rm_field // mod=00
            } else if self.disp >= -128 && self.disp <= 127 {
                0x40 | reg_field | rm_field // mod=01 (disp8)
            } else {
                0x80 | reg_field | rm_field // mod=10 (disp32)
            }
        } else {
            0xC0 | reg_field | rm_field // mod=11 (register)
        }
    }

    /// Encode SIB byte if needed
    fn encode_sib(&self) -> Option<u8> {
        if !self.is_mem || self.index.is_none() {
            return None;
        }

        let index = self.index.unwrap() & 0x07;
        let base = self.rm & 0x07;
        let scale_bits = match self.scale {
            1 => 0x00,
            2 => 0x40,
            4 => 0x80,
            8 => 0xC0,
            _ => 0x00,
        };

        Some(scale_bits | (index << 3) | base)
    }
}

impl Default for EvexPrefix {
    fn default() -> Self {
        Self::new()
    }
}

/// AVX-512 instruction encoder
pub struct Avx512Encoder {
    buffer: Vec<u8>,
}

impl Avx512Encoder {
    pub fn new() -> Self {
        Self { buffer: Vec::new() }
    }

    /// VMOVDQU64 zmm, [base + index*8 + disp] - Load 512 bits
    /// Opcode: EVEX.512.F3.0F.W1 6F /r
    pub fn vmovdqu64_load(&mut self, dest_zmm: u8, base: u8, index: u8, disp: i32) {
        let prefix = EvexPrefix::new()
            .with_dest(dest_zmm)
            .with_mem_base(base)
            .with_index(index, 8)
            .with_disp(disp)
            .with_map(0x01); // 0F map

        // EVEX prefix with pp=10 (F3)
        let mut evex = prefix.encode_prefix();
        evex[2] = (evex[2] & 0xFC) | 0x02; // pp=10 for F3

        self.buffer.extend_from_slice(&evex);
        self.buffer.push(0x6F); // opcode
        self.buffer.push(prefix.encode_modrm());

        if let Some(sib) = prefix.encode_sib() {
            self.buffer.push(sib);
        }

        // Displacement (64-byte granularity for disp8)
        if disp != 0 {
            if disp >= -128 * 64 && disp <= 127 * 64 && disp % 64 == 0 {
                self.buffer.push((disp / 64) as u8);
            } else {
                self.buffer.extend_from_slice(&(disp as i32).to_le_bytes());
            }
        }
    }

    /// VMOVDQU64 [base + index*8 + disp], zmm - Store 512 bits
    /// Opcode: EVEX.512.F3.0F.W1 7F /r
    pub fn vmovdqu64_store(&mut self, base: u8, index: u8, src_zmm: u8, disp: i32) {
        let prefix = EvexPrefix::new()
            .with_dest(src_zmm)
            .with_mem_base(base)
            .with_index(index, 8)
            .with_disp(disp)
            .with_map(0x01);

        let mut evex = prefix.encode_prefix();
        evex[2] = (evex[2] & 0xFC) | 0x02; // pp=10 for F3

        self.buffer.extend_from_slice(&evex);
        self.buffer.push(0x7F); // opcode (store)
        self.buffer.push(prefix.encode_modrm());

        if let Some(sib) = prefix.encode_sib() {
            self.buffer.push(sib);
        }

        if disp != 0 {
            if disp >= -128 * 64 && disp <= 127 * 64 && disp % 64 == 0 {
                self.buffer.push((disp / 64) as u8);
            } else {
                self.buffer.extend_from_slice(&(disp as i32).to_le_bytes());
            }
        }
    }

    /// VPADDQ zmm, zmm, zmm - Add packed 64-bit integers
    /// Opcode: EVEX.512.66.0F.W1 D4 /r
    pub fn vpaddq_zmm(&mut self, dest: u8, src1: u8, src2: u8) {
        let prefix = EvexPrefix::new()
            .with_dest(dest)
            .with_src1(src1)
            .with_src2_reg(src2)
            .with_map(0x01);

        self.buffer.extend_from_slice(&prefix.encode_prefix());
        self.buffer.push(0xD4); // opcode
        self.buffer.push(prefix.encode_modrm());
    }

    /// VPXORQ zmm, zmm, zmm - XOR packed 64-bit integers (zero registers)
    /// Opcode: EVEX.512.66.0F.W1 EF /r
    pub fn vpxorq_zmm(&mut self, dest: u8, src1: u8, src2: u8) {
        let prefix = EvexPrefix::new()
            .with_dest(dest)
            .with_src1(src1)
            .with_src2_reg(src2)
            .with_map(0x01);

        self.buffer.extend_from_slice(&prefix.encode_prefix());
        self.buffer.push(0xEF); // opcode
        self.buffer.push(prefix.encode_modrm());
    }

    /// Get the encoded bytes
    pub fn finalize(self) -> Vec<u8> {
        self.buffer
    }

    /// Get current buffer
    pub fn as_bytes(&self) -> &[u8] {
        &self.buffer
    }

    /// Append raw bytes
    pub fn emit_bytes(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }
}

impl Default for Avx512Encoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evex_prefix() {
        let prefix = EvexPrefix::new().with_dest(0).with_src1(1).with_src2_reg(2);

        let bytes = prefix.encode_prefix();
        assert_eq!(bytes[0], 0x62, "EVEX identifier");
        println!(
            "EVEX prefix: {:02X} {:02X} {:02X} {:02X}",
            bytes[0], bytes[1], bytes[2], bytes[3]
        );
    }

    #[test]
    fn test_vpaddq_zmm() {
        let mut enc = Avx512Encoder::new();
        enc.vpaddq_zmm(0, 1, 2); // zmm0 = zmm1 + zmm2

        let bytes = enc.finalize();
        println!("VPADDQ zmm0, zmm1, zmm2: {:02X?}", bytes);
        assert!(!bytes.is_empty());
    }
}
