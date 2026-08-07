// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Conformance tests for every [`InputElement`](crate::scalar_fn::InputElement) in this crate.

use std::sync::Arc;

use vortex_buffer::BitBuffer;
use vortex_buffer::ByteBuffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use crate::IntoArray;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::BoolArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinViewArray;
use crate::arrays::varbinview::BinaryView;
use crate::dtype::DType;
use crate::dtype::Nullability;
use crate::scalar_fn::assert_element_conforms;
use crate::scalar_fn::row::tests::TestBytes;
use crate::validity::Validity;

/// A `Utf8` column whose single null row carries a view naming a buffer that does not exist, at
/// an offset far past the end of the data. Reading its _bytes_ densely panics; reading its
/// _length_ does not, which is exactly the distinction `DENSE_SAFE` encodes.
fn hostile_views() -> VortexResult<crate::ArrayRef> {
    let views = buffer![
        BinaryView::make_view(b"a longer string here", 0, 0),
        BinaryView::new_ref(64, *b"junk", 9, 4096),
    ];
    Ok(VarBinViewArray::try_new(
        views,
        Arc::from([ByteBuffer::copy_from(b"a longer string here")]),
        DType::Utf8(Nullability::Nullable),
        Validity::from_iter([true, false]),
    )?
    .into_array())
}

#[test]
fn primitive_element_conforms() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    // The extremes sit at the rows that are then marked null.
    let array = PrimitiveArray::new(
        buffer![i32::MAX, 1, i32::MIN, 2],
        Validity::from_iter([false, true, false, true]),
    )
    .into_array();

    assert_element_conforms::<i32>(array, &DType::Utf8(Nullability::NonNullable), &mut ctx)
}

#[test]
fn bool_element_conforms() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let array = BoolArray::new(
        BitBuffer::from(vec![true, true, false, true]),
        Validity::from_iter([false, true, true, false]),
    )
    .into_array();

    assert_element_conforms::<bool>(array, &DType::Utf8(Nullability::NonNullable), &mut ctx)
}

#[test]
fn test_bytes_element_conforms() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    assert_element_conforms::<TestBytes>(
        hostile_views()?,
        &DType::Bool(Nullability::NonNullable),
        &mut ctx,
    )
}
