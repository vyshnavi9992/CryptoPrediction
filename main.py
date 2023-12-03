import streamlit as st
import streamlit_authenticator as stauth
from dependencies import sign_up, fetch_users


st.set_page_config(page_title='Crypto App', page_icon='💰', initial_sidebar_state='collapsed')

st.subheader("📈 Crypto Prediction Web Application")

st.sidebar.image("data/bitcoin_logo.jpg",caption="Menu Options")


try:
    users = fetch_users()
    emails = []
    usernames = []
    passwords = []

    for user in users:
        emails.append(user['key'])
        usernames.append(user['username'])
        passwords.append(user['password'])

    credentials = {'usernames': {}}
    for index in range(len(emails)):
        credentials['usernames'][usernames[index]] = {'name': emails[index], 'password': passwords[index]}
    
    Authenticator = stauth.Authenticate(credentials, cookie_name='Streamlit', key='abcdef', cookie_expiry_days=4)

    email, authentication_status, username = Authenticator.login(':green[Login]', 'main')

    info, info1 = st.columns(2)

    if not authentication_status:
        sign_up()

    if username:
        if username in usernames:
            if authentication_status:
                # let User see app
                st.sidebar.subheader(f'Welcome {username}')
                Authenticator.logout('Log Out', 'sidebar')
                st.subheader('This is the home page')
                st.subheader('Crypto Web App📈   A complete solution for all your crypto related decision built on powerful Neural Network and Time Series Forecasting')
                st.write("[See more >](https://github.com/vyshnavi9992/CryptoPrediction)")
                st.markdown(
                    """
                    ---
                    Created with ❤️ by Vysh
                    
                    """
                )

            elif not authentication_status:
                with info:
                    st.error('Incorrect Password or username')
            else:
                with info:
                    st.warning('Please feed in your credentials')
        else:
            with info:
                st.warning('Username does not exist, Please Sign up')


except:
    st.success('Refresh Page')