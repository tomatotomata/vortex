// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors
#![allow(clippy::clone_on_ref_ptr)]

use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::assert_arrays_eq;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::validity::Validity;

#[test]
pub fn varbin_view() {
    let mut ctx = array_session().create_execution_ctx();
    let binary_arr =
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"]);
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"]),
        &mut ctx
    );
}

#[test]
pub fn slice_array() {
    let mut ctx = array_session().create_execution_ctx();
    let binary_arr =
        VarBinViewArray::from_iter_str(["hello world", "hello world this is a long string"])
            .slice(1..2)
            .unwrap();
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["hello world this is a long string"]),
        &mut ctx
    );
}

#[test]
pub fn flatten_array() {
    let mut ctx = array_session().create_execution_ctx();
    let binary_arr = VarBinViewArray::from_iter_str(["string1", "string2"]);
    assert_arrays_eq!(
        binary_arr,
        VarBinViewArray::from_iter_str(["string1", "string2"]),
        &mut ctx
    );
}

#[test]
pub fn binary_view_size_and_alignment() {
    assert_eq!(size_of::<BinaryView>(), 16);
    assert_eq!(align_of::<BinaryView>(), 16);
}

// Null views in VarBinView are validated
#[test]
pub fn binary_view_null_view() {
    let views = Buffer::<BinaryView>::copy_from(vec![
        BinaryView::new_inlined(b"ololo"),
        BinaryView::new_ref(14, *b"hell", 0, 0),
        BinaryView::new_ref(13, *b"AAAA", 0xDEAD_BEEF, 0xF000_0000),
    ]);
    let data = b"hello world ololo";

    let validity = BitBuffer::from_iter([true, true, false]);
    let validity = BoolArray::new(validity, Validity::NonNullable);
    let validity = Validity::Array(validity.into_array());
    let buffers = Arc::new([ByteBuffer::from(data.to_vec())]);
    let dtype = DType::Utf8(Nullability::Nullable);

    let array = VarBinViewArray::try_new(views.clone(), buffers.clone(), dtype.clone(), validity);
    assert!(array.is_err());
    let array = VarBinViewArray::try_new(views, buffers, dtype, Validity::AllInvalid);
    assert!(array.is_err());
}

/// Validation of Null views in VarBinView doesn't check prefix and contents
#[test]
pub fn binary_view_null_view_in_bounds() -> VortexResult<()> {
    let data = b"hello world foo\xFF\xFE\xFD";
    let buffers = Arc::new([ByteBuffer::from(data.to_vec())]);
    let dtype = DType::Utf8(Nullability::Nullable);

    let validity = BoolArray::new(BitBuffer::from_iter([true, false]), Validity::NonNullable);
    let validity = Validity::Array(validity.into_array());

    let valid_row = BinaryView::new_ref(15, *b"hell", 0, 0);
    let garbage_row = BinaryView::new_ref(13, *b"XXXX", 0, 5);

    let views = Buffer::<BinaryView>::copy_from(vec![valid_row, garbage_row]);
    VarBinViewArray::try_new(views.clone(), buffers.clone(), dtype.clone(), validity)?;
    VarBinViewArray::try_new(views, buffers, dtype, Validity::AllInvalid)?;
    Ok(())
}
