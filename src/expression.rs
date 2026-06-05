use std::fmt::{Debug, Formatter};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use fnv::FnvHashMap as HashMap;

#[cfg(feature = "enable_quadratic")]
use crate::IntoQuadraticExpression;
use crate::affine_expression_trait::IntoAffineExpression;
use crate::constraint;
use crate::variable::{FormatWithVars, Variable};
use crate::{Constraint, Solution};

/// Represents a pair of variables in a quadratic term.
///
/// The pair is ordered so that `x*y` and `y*x` map to the same key.
#[cfg(feature = "enable_quadratic")]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VariablePair {
    /// First variable in the pair
    pub var1: Variable,
    /// Second variable in the pair
    pub var2: Variable,
}

#[cfg(feature = "enable_quadratic")]
impl VariablePair {
    /// Create a new variable pair, ensuring consistent ordering for commutativity (x*y = y*x)
    pub fn new(var1: Variable, var2: Variable) -> Self {
        if var1.index() <= var2.index() {
            VariablePair { var1, var2 }
        } else {
            VariablePair {
                var1: var2,
                var2: var1,
            }
        }
    }
}

/// A pure quadratic expression containing only quadratic terms (similar to [LinearExpression])
#[cfg(feature = "enable_quadratic")]
#[derive(Clone)]
pub struct QuadraticExpression {
    pub(crate) coefficients: HashMap<VariablePair, f64>,
}

#[cfg(feature = "enable_quadratic")]
impl QuadraticExpression {
    /// Create a new empty quadratic expression
    pub fn new() -> Self {
        QuadraticExpression {
            coefficients: HashMap::default(),
        }
    }

    /// Add a quadratic term
    pub fn add_quadratic_term(&mut self, var1: Variable, var2: Variable, coefficient: f64) {
        let pair = VariablePair::new(var1, var2);
        *self.coefficients.entry(pair).or_default() += coefficient;
    }

    /// Evaluate the quadratic terms with given variable values
    pub fn eval_with<S: Solution>(&self, values: &S) -> f64 {
        self.coefficients
            .iter()
            .map(|(pair, &coeff)| coeff * values.value(pair.var1) * values.value(pair.var2))
            .sum()
    }
}

#[cfg(feature = "enable_quadratic")]
impl Default for QuadraticExpression {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "enable_quadratic")]
impl Debug for QuadraticExpression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuadraticExpression")
            .field("coefficients", &self.coefficients)
            .finish()
    }
}

#[cfg(feature = "enable_quadratic")]
impl FormatWithVars for QuadraticExpression {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, mut variable_format: FUN) -> std::fmt::Result
    where
        FUN: FnMut(&mut Formatter<'_>, Variable) -> std::fmt::Result,
    {
        let mut first = true;
        for (pair, &coeff) in &self.coefficients {
            if coeff != 0f64 {
                if first {
                    first = false;
                } else {
                    write!(f, " + ")?;
                }
                if (coeff - 1.).abs() > f64::EPSILON {
                    write!(f, "{} ", coeff)?;
                }
                variable_format(f, pair.var1)?;
                write!(f, "*")?;
                variable_format(f, pair.var2)?;
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

/// An linear expression without a constant component
pub struct LinearExpression {
    pub(crate) coefficients: HashMap<Variable, f64>,
}

impl IntoAffineExpression for LinearExpression {
    type Iter = std::collections::hash_map::IntoIter<Variable, f64>;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        self.coefficients.into_iter()
    }
}

/// Return type for `&'a LinearExpression::linear_coefficients`
#[doc(hidden)]
pub struct CopiedCoefficients<'a>(std::collections::hash_map::Iter<'a, Variable, f64>);

impl Iterator for CopiedCoefficients<'_> {
    type Item = (Variable, f64);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|(&var, &c)| (var, c))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl<'a> IntoAffineExpression for &'a LinearExpression {
    type Iter = CopiedCoefficients<'a>;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        CopiedCoefficients(self.coefficients.iter())
    }
}

impl FormatWithVars for LinearExpression {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, mut variable_format: FUN) -> std::fmt::Result
    where
        FUN: FnMut(&mut Formatter<'_>, Variable) -> std::fmt::Result,
    {
        let mut first = true;
        for (&var, &coeff) in &self.coefficients {
            if coeff != 0f64 {
                if first {
                    first = false;
                } else {
                    write!(f, " + ")?;
                }
                if (coeff - 1.).abs() > f64::EPSILON {
                    write!(f, "{} ", coeff)?;
                }
                variable_format(f, var)?;
            }
        }
        if first {
            write!(f, "0")?;
        }
        Ok(())
    }
}

/// Represents an affine expression, such as `2x + 3` or `x + y + z`.
///
/// With the `enable_quadratic` feature it can also represent quadratic
/// expressions, such as `x^2 + y^2 + xy + 2x + 3`.
pub struct Expression {
    #[cfg(feature = "enable_quadratic")]
    pub(crate) quadratic: QuadraticExpression,
    pub(crate) linear: LinearExpression,
    pub(crate) constant: f64,
}

impl IntoAffineExpression for Expression {
    type Iter = <LinearExpression as IntoAffineExpression>::Iter;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        #[cfg(feature = "enable_quadratic")]
        assert!(
            self.is_affine(),
            "Cannot convert a quadratic expression into an affine expression: it contains quadratic terms"
        );
        self.linear.linear_coefficients()
    }

    #[inline]
    fn constant(&self) -> f64 {
        self.constant
    }
}

/// This implementation copies all the variables and coefficients from the referenced
/// Expression into the created iterator
impl<'a> IntoAffineExpression for &'a Expression {
    type Iter = <&'a LinearExpression as IntoAffineExpression>::Iter;

    #[inline]
    fn linear_coefficients(self) -> Self::Iter {
        #[cfg(feature = "enable_quadratic")]
        assert!(
            self.is_affine(),
            "Cannot convert a quadratic expression into an affine expression: it contains quadratic terms"
        );
        (&self.linear).linear_coefficients()
    }

    #[inline]
    fn constant(&self) -> f64 {
        self.constant
    }
}

impl PartialEq for Expression {
    fn eq(&self, other: &Self) -> bool {
        let affine_eq = self.constant.eq(&other.constant)
            && self.linear.coefficients.eq(&other.linear.coefficients);
        #[cfg(feature = "enable_quadratic")]
        let affine_eq = affine_eq
            && self
                .quadratic
                .coefficients
                .eq(&other.quadratic.coefficients);
        affine_eq
    }
}

impl Clone for Expression {
    fn clone(&self) -> Self {
        Expression {
            #[cfg(feature = "enable_quadratic")]
            quadratic: self.quadratic.clone(),
            linear: LinearExpression {
                coefficients: self.linear.coefficients.clone(),
            },
            constant: self.constant,
        }
    }
}

impl Debug for Expression {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.format_debug(f)
    }
}

impl Default for Expression {
    fn default() -> Self {
        Expression::from(0.)
    }
}

impl Expression {
    /// Create an expression that has the value 0, but has memory allocated
    /// for `capacity` coefficients.
    pub fn with_capacity(capacity: usize) -> Self {
        Expression {
            #[cfg(feature = "enable_quadratic")]
            quadratic: QuadraticExpression::new(),
            linear: LinearExpression {
                coefficients: HashMap::with_capacity_and_hasher(capacity, Default::default()),
            },
            constant: 0.0,
        }
    }

    /// Create a concrete expression struct from anything that has linear coefficients and a constant
    ///
    /// ```
    /// # use good_lp::Expression;
    /// Expression::from_other_affine(0.); // A constant expression
    /// ```
    pub fn from_other_affine<E: IntoAffineExpression>(source: E) -> Self {
        source.into_expression()
    }

    /// Create a concrete expression struct from anything that can be expressed as a quadratic expression
    #[cfg(feature = "enable_quadratic")]
    pub fn from_other_quadratic<E: IntoQuadraticExpression>(source: E) -> Self {
        source.into_quadratic_expression()
    }

    /// Creates a constraint indicating that this expression
    /// is lesser than or equal to the right hand side
    pub fn leq<RHS>(self, rhs: RHS) -> Constraint
    where
        Expression: Sub<RHS, Output = Expression>,
    {
        constraint::leq(self, rhs)
    }

    /// Creates a constraint indicating that this expression
    /// is greater than or equal to the right hand side
    pub fn geq<RHS: Sub<Expression, Output = Expression>>(self, rhs: RHS) -> Constraint {
        constraint::geq(self, rhs)
    }

    /// Creates a constraint indicating that this expression
    /// is equal to the right hand side
    pub fn eq<RHS>(self, rhs: RHS) -> Constraint
    where
        Expression: Sub<RHS, Output = Expression>,
    {
        constraint::eq(self, rhs)
    }

    /// Performs self = self + (a * b)
    #[inline]
    pub fn add_mul<N: Into<f64>, E: IntoAffineExpression>(&mut self, a: N, b: E) {
        let factor = a.into();
        let constant = b.constant();
        for (var, value) in b.linear_coefficients().into_iter() {
            *self.linear.coefficients.entry(var).or_default() += factor * value
        }
        self.constant += factor * constant;
    }

    /// Add a quadratic term to this expression
    #[cfg(feature = "enable_quadratic")]
    pub fn add_quadratic_term(&mut self, var1: Variable, var2: Variable, coefficient: f64) {
        self.quadratic.add_quadratic_term(var1, var2, coefficient);
    }

    /// Add a linear term to this expression
    #[cfg(feature = "enable_quadratic")]
    pub fn add_linear_term(&mut self, var: Variable, coefficient: f64) {
        *self.linear.coefficients.entry(var).or_default() += coefficient;
    }

    /// Add a constant term to this expression
    #[cfg(feature = "enable_quadratic")]
    pub fn add_constant(&mut self, value: f64) {
        self.constant += value;
    }

    /// Returns true if this expression contains no quadratic terms (always true
    /// unless the `enable_quadratic` feature is enabled).
    #[inline]
    pub fn is_affine(&self) -> bool {
        #[cfg(feature = "enable_quadratic")]
        {
            self.quadratic
                .coefficients
                .values()
                .all(|&coeff| coeff.abs() < f64::EPSILON)
        }
        #[cfg(not(feature = "enable_quadratic"))]
        {
            true
        }
    }

    /// See [IntoAffineExpression::eval_with]
    pub fn eval_with<S: Solution>(&self, values: &S) -> f64 {
        #[cfg(feature = "enable_quadratic")]
        {
            // Evaluate linear terms manually to avoid the affine-conversion assert.
            let mut result = self.constant;
            for (&var, &coeff) in &self.linear.coefficients {
                result += coeff * values.value(var);
            }
            result + self.quadratic.eval_with(values)
        }
        #[cfg(not(feature = "enable_quadratic"))]
        {
            IntoAffineExpression::eval_with(self, values)
        }
    }
}

#[inline]
pub fn add_mul<LHS: Into<Expression>, RHS: IntoAffineExpression>(
    lhs: LHS,
    rhs: RHS,
    factor: f64,
) -> Expression {
    let mut result = lhs.into();
    result.add_mul(factor, rhs);
    result
}

#[inline]
pub fn sub<LHS: Into<Expression>, RHS: IntoAffineExpression>(lhs: LHS, rhs: RHS) -> Expression {
    add_mul(lhs, rhs, -1.)
}

#[inline]
pub fn add<LHS: Into<Expression>, RHS: IntoAffineExpression>(lhs: LHS, rhs: RHS) -> Expression {
    add_mul(lhs, rhs, 1.)
}

#[cfg(not(feature = "enable_quadratic"))]
impl FormatWithVars for Expression {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, variable_format: FUN) -> std::fmt::Result
    where
        FUN: FnMut(&mut Formatter<'_>, Variable) -> std::fmt::Result,
    {
        self.linear.format_with(f, variable_format)?;
        if self.constant.abs() >= f64::EPSILON {
            write!(f, " + {}", self.constant)?;
        }
        Ok(())
    }
}

#[cfg(feature = "enable_quadratic")]
impl FormatWithVars for Expression {
    fn format_with<FUN>(&self, f: &mut Formatter<'_>, mut variable_format: FUN) -> std::fmt::Result
    where
        FUN: FnMut(&mut Formatter<'_>, Variable) -> std::fmt::Result,
    {
        let mut has_terms = false;

        for (&var, &coeff) in &self.linear.coefficients {
            if coeff != 0f64 {
                if has_terms {
                    write!(f, " + ")?;
                }
                if (coeff - 1.).abs() > f64::EPSILON {
                    write!(f, "{} ", coeff)?;
                }
                variable_format(f, var)?;
                has_terms = true;
            }
        }

        for (pair, &coeff) in &self.quadratic.coefficients {
            if coeff != 0f64 {
                if has_terms {
                    write!(f, " + ")?;
                }
                if (coeff - 1.).abs() > f64::EPSILON {
                    write!(f, "{} ", coeff)?;
                }
                variable_format(f, pair.var1)?;
                write!(f, "*")?;
                variable_format(f, pair.var2)?;
                has_terms = true;
            }
        }

        if self.constant.abs() >= f64::EPSILON {
            if has_terms {
                write!(f, " + {}", self.constant)?;
            } else {
                write!(f, "{}", self.constant)?;
            }
        } else if !has_terms {
            write!(f, "0")?;
        }

        Ok(())
    }
}

impl<RHS: IntoAffineExpression> SubAssign<RHS> for Expression {
    #[inline]
    fn sub_assign(&mut self, rhs: RHS) {
        self.add_mul(-1., rhs)
    }
}

impl<RHS: IntoAffineExpression> AddAssign<RHS> for Expression {
    #[inline]
    fn add_assign(&mut self, rhs: RHS) {
        self.add_mul(1, rhs);
    }
}

impl Neg for Expression {
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self::Output {
        self *= -1;
        self
    }
}

impl<N: Into<f64>> MulAssign<N> for Expression {
    #[inline]
    fn mul_assign(&mut self, rhs: N) {
        let factor = rhs.into();
        #[cfg(feature = "enable_quadratic")]
        for value in self.quadratic.coefficients.values_mut() {
            *value *= factor
        }
        for value in self.linear.coefficients.values_mut() {
            *value *= factor
        }
        self.constant *= factor
    }
}

impl<N: Into<f64>> Mul<N> for Expression {
    type Output = Expression;

    #[inline]
    fn mul(mut self, rhs: N) -> Self::Output {
        self.mul_assign(rhs);
        self
    }
}

impl<N: Into<f64>> Div<N> for Expression {
    type Output = Expression;

    #[inline]
    fn div(mut self, rhs: N) -> Self::Output {
        self.mul_assign(1. / rhs.into());
        self
    }
}

macro_rules! impl_mul {
    ($($t:ty),*) =>{$(
        impl Mul<Expression> for $t {
            type Output = Expression;

            fn mul(self, mut rhs: Expression) -> Self::Output {
                rhs *= self;
                rhs
            }
        }
    )*}
}
impl_mul!(f64, i32);

// Without the `enable_quadratic` feature, `Expression` and `Variable` can be added to /
// subtracted from any affine expression through a single blanket impl.
#[cfg(not(feature = "enable_quadratic"))]
macro_rules! impl_ops_local {
    ($( $typename:ident : $([generic $generic:ident])? $other:ident),*) => {$(
        impl<$($generic: IntoAffineExpression)?> Sub<$other>
        for $typename {
            type Output = Expression;
            fn sub(self, rhs: $other) -> Self::Output { sub(self, rhs) }
        }

        impl<$($generic: IntoAffineExpression)?> Add<$other>
        for $typename {
            type Output = Expression;
            fn add(self, rhs: $other) -> Self::Output { add(self, rhs) }
        }
    )*}
}

#[cfg(not(feature = "enable_quadratic"))]
impl_ops_local!(
    Expression: [generic RHS] RHS,
    Variable: [generic RHS] RHS,
    f64: Expression,
    f64: Variable,
    i32: Expression,
    i32: Variable
);

// With the `enable_quadratic` feature, the blanket `impl<RHS: IntoAffineExpression>` used in
// the non-quadratic build cannot be used: it would overlap with the quadratic-aware
// `Add<Expression> for Expression` below (because `Expression: IntoAffineExpression`). Instead
// we keep a *single* blanket, bound on `AffineOperand` -- every affine operand except
// `Expression`/`&Expression`, which carry quadratic terms and are handled explicitly. Using one
// blanket (rather than one impl per numeric type) preserves main's integer-literal inference,
// so no call sites need type annotations.
#[cfg(feature = "enable_quadratic")]
#[doc(hidden)]
pub trait AffineOperand: IntoAffineExpression {}

#[cfg(feature = "enable_quadratic")]
macro_rules! impl_affine_operand {
    ($($t:ty),*) => {$( impl AffineOperand for $t {} )*};
}
#[cfg(feature = "enable_quadratic")]
impl_affine_operand!(
    f64,
    f32,
    u32,
    u16,
    u8,
    i32,
    i16,
    i8,
    Variable,
    LinearExpression
);
#[cfg(feature = "enable_quadratic")]
impl AffineOperand for &Variable {}
#[cfg(feature = "enable_quadratic")]
impl AffineOperand for &LinearExpression {}
#[cfg(feature = "enable_quadratic")]
impl AffineOperand for Option<Variable> {}

#[cfg(feature = "enable_quadratic")]
macro_rules! impl_ops_local_quadratic {
    ($($typename:ty),*) => {$(
        impl<RHS: AffineOperand> Add<RHS> for $typename {
            type Output = Expression;
            fn add(self, rhs: RHS) -> Self::Output { add(self, rhs) }
        }
        impl<RHS: AffineOperand> Sub<RHS> for $typename {
            type Output = Expression;
            fn sub(self, rhs: RHS) -> Self::Output { sub(self, rhs) }
        }
    )*};
}
#[cfg(feature = "enable_quadratic")]
impl_ops_local_quadratic!(Expression, Variable);

// `Expression op Expression` additionally merges the right-hand side's quadratic terms.
#[cfg(feature = "enable_quadratic")]
impl Add<Expression> for Expression {
    type Output = Expression;
    fn add(mut self, rhs: Expression) -> Self::Output {
        for (pair, coeff) in rhs.quadratic.coefficients {
            self.add_quadratic_term(pair.var1, pair.var2, coeff);
        }
        for (var, coeff) in rhs.linear.coefficients {
            self.add_linear_term(var, coeff);
        }
        self.add_constant(rhs.constant);
        self
    }
}

#[cfg(feature = "enable_quadratic")]
impl Sub<Expression> for Expression {
    type Output = Expression;
    fn sub(mut self, rhs: Expression) -> Self::Output {
        for (pair, coeff) in rhs.quadratic.coefficients {
            self.add_quadratic_term(pair.var1, pair.var2, -coeff);
        }
        for (var, coeff) in rhs.linear.coefficients {
            self.add_linear_term(var, -coeff);
        }
        self.add_constant(-rhs.constant);
        self
    }
}

// `Variable op Expression` routes through `Expression` so the right-hand side's quadratic
// terms survive (the `AffineOperand` blanket above only covers affine right-hand sides).
#[cfg(feature = "enable_quadratic")]
impl Add<Expression> for Variable {
    type Output = Expression;
    fn add(self, rhs: Expression) -> Self::Output {
        Expression::from(self) + rhs
    }
}
#[cfg(feature = "enable_quadratic")]
impl Sub<Expression> for Variable {
    type Output = Expression;
    fn sub(self, rhs: Expression) -> Self::Output {
        Expression::from(self) - rhs
    }
}

// `number op Expression` / `number op Variable`, matching the `f64`/`i32` left-hand-side set
// of the non-quadratic `impl_ops_local!`.
#[cfg(feature = "enable_quadratic")]
macro_rules! impl_num_lhs_quadratic {
    ($($num:ty),*) => {$(
        impl Add<Expression> for $num {
            type Output = Expression;
            fn add(self, rhs: Expression) -> Self::Output { rhs + self }
        }
        impl Sub<Expression> for $num {
            type Output = Expression;
            fn sub(self, rhs: Expression) -> Self::Output { Expression::from(self) - rhs }
        }
        impl Add<Variable> for $num {
            type Output = Expression;
            fn add(self, rhs: Variable) -> Self::Output { Expression::from(self) + rhs }
        }
        impl Sub<Variable> for $num {
            type Output = Expression;
            fn sub(self, rhs: Variable) -> Self::Output { Expression::from(self) - rhs }
        }
    )*};
}
#[cfg(feature = "enable_quadratic")]
impl_num_lhs_quadratic!(f64, i32);

// Multiplication of two affine expressions creates quadratic terms.
#[cfg(feature = "enable_quadratic")]
impl Mul<Variable> for Variable {
    type Output = Expression;
    fn mul(self, rhs: Variable) -> Self::Output {
        let mut expr = Expression::default();
        expr.add_quadratic_term(self, rhs, 1.0);
        expr
    }
}

#[cfg(feature = "enable_quadratic")]
impl Mul<Variable> for Expression {
    type Output = Expression;
    fn mul(self, rhs: Variable) -> Self::Output {
        let mut result = Expression::default();
        for (var, coeff) in self.linear.coefficients {
            result.add_quadratic_term(var, rhs, coeff);
        }
        if self.constant.abs() >= f64::EPSILON {
            result.add_linear_term(rhs, self.constant);
        }
        result
    }
}

#[cfg(feature = "enable_quadratic")]
impl Mul<Expression> for Variable {
    type Output = Expression;
    fn mul(self, rhs: Expression) -> Self::Output {
        rhs * self
    }
}

#[cfg(feature = "enable_quadratic")]
impl Mul<Expression> for Expression {
    type Output = Expression;
    fn mul(self, rhs: Expression) -> Self::Output {
        let mut result = Expression::default();
        // Linear × linear → quadratic
        for (&var1, &coeff1) in &self.linear.coefficients {
            for (&var2, &coeff2) in &rhs.linear.coefficients {
                result.add_quadratic_term(var1, var2, coeff1 * coeff2);
            }
        }
        // Linear × constant → linear
        if rhs.constant.abs() >= f64::EPSILON {
            for (&var, &coeff) in &self.linear.coefficients {
                result.add_linear_term(var, coeff * rhs.constant);
            }
        }
        if self.constant.abs() >= f64::EPSILON {
            for (&var, &coeff) in &rhs.linear.coefficients {
                result.add_linear_term(var, self.constant * coeff);
            }
        }
        // Constant × constant → constant
        if self.constant.abs() >= f64::EPSILON && rhs.constant.abs() >= f64::EPSILON {
            result.add_constant(self.constant * rhs.constant);
        }
        result
    }
}

macro_rules! impl_conv {
    ( $( $typename:ident ),* ) => {$(
        impl From<$typename> for Expression {
            fn from(x: $typename) -> Expression { Expression::from_other_affine(x) }
        }
    )*}
}
impl_conv!(f64, i32, Variable);

impl<E: IntoAffineExpression> std::iter::Sum<E> for Expression {
    fn sum<I: Iterator<Item = E>>(iter: I) -> Self {
        let (capacity, _) = iter.size_hint();
        let mut res = Expression::with_capacity(capacity);
        for i in iter {
            res.add_assign(i)
        }
        res
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::variables;

    #[test]
    fn expression_manipulation() {
        variables! {vars: v0; v1; }
        assert_eq!((3. - v0) - v1, (-1.) * v0 + (-1.) * v1 + 3.)
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn eval() {
        let mut vars = variables!();
        let a = vars.add_variable();
        let b = vars.add_variable();
        let mut values = HashMap::new();
        values.insert(a, 100);
        values.insert(b, -1);
        assert_eq!((a + 3 * (b + 3)).eval_with(&values), 106.)
    }

    #[cfg(feature = "enable_quadratic")]
    #[allow(clippy::float_cmp)]
    #[test]
    fn expression_manipulation_quadratic() {
        variables! {vars: v0; v1; }
        let expression_0 = (3.0_f64 - v0) - v1 * v1;
        let expression_1 = (-1.0) * v0 + (-1.0) * v1 * v1 + 3.0;
        let expression_0_str = format!("{:?}", expression_0);
        assert_eq!(expression_0_str, "-1 v0 + -1 v1*v1 + 3");
        assert_eq!(expression_0, expression_1);
        let values = HashMap::from([(v0, 100.), (v1, -1.)]);
        assert_eq!(expression_0.eval_with(&values), -98.0)
    }
}
