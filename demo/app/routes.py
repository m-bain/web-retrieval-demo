from flask import render_template, flash, redirect, request, url_for, session
from demo.app import app
from demo.app.forms import LoginForm, QueryForm
from frozen_client import process_text_query
import json


@app.route('/')
@app.route('/index')
def index():
    user = {'username': 'Max'}
    posts = [
        {
            'author': {'username': 'John'},
            'body': 'Beautiful day in Portland!'
        },
        {
            'author': {'username': 'Susan'},
            'body': 'The Avengers movie was so cool!'
        }
    ]
    return render_template('index.html', title='Home', user=user, posts=posts)


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        flash('Login requested for user {}, remember_me={}'.format(
            form.username.data, form.remember_me.data))
        return redirect('/index')
    return render_template('login.html', title='Sign In', form=form)


@app.route('/search', methods=['GET', 'POST'])
def search():
    form = QueryForm()
    if form.validate_on_submit():
        # flash('Login requested for user {}, remember_me={}'.format(
        #    form.username.data, form.remember_me.data))
        session['query'] = form.query.data
        session['topk'] = form.topk.data
        return redirect(url_for('results'))
    return render_template('search.html', title='Search', form=form)


@app.route('/results', methods=['GET'])
def results():
    query = session['query']
    topk = int(session['topk'])
    results_json = process_text_query(
        text_query=query,
        topk=topk
    )

    vid_data = results_json
    return render_template('results.html', title='Results', query=query, topk=topk,
                           vid_data=vid_data)


@app.route('/query')
def query():
    query = request.args['query']
    topk = int(request.args['topk'])
    results_json = process_text_query(
        text_query=query,
        topk=topk
    )
    return json.dumps(results_json)
