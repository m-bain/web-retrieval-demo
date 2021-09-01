from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired, NumberRange


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')


class QueryForm(FlaskForm):
    query = StringField('Enter text query:', validators=[DataRequired()])
    topk = SelectField('Display top k results:', choices=[4, 8, 16])
    submit = SubmitField('Search')
