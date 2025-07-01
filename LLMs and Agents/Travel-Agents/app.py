# pylint: disable = invalid-name
import streamlit as st
from langchain_core.messages import HumanMessage

from agent import Agent


def initialize_agent():
    if 'agent' not in st.session_state:
        st.session_state.agent = Agent()


def render_custom_css():
    st.markdown(
        '''
        <style>
        .main-title {
            font-size: 2.5em;
            color: #333;
            text-align: center;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        .sub-title {
            font-size: 1.2em;
            color: #333;
            text-align: left;
            margin-bottom: 0.5em;
        }
        .center-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            width: 100%;
        }
        .query-box {
            width: 80%;
            max-width: 600px;
            margin-top: 0.5em;
            margin-bottom: 1em;
        }
        .query-container {
            width: 80%;
            max-width: 600px;
            margin: 0 auto;
        }
        </style>
        ''', unsafe_allow_html=True)


def render_ui():
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    st.markdown('<div class="main-title">✈️🌍 AI Travel Agent 🏨🗺️</div>', unsafe_allow_html=True)
    st.markdown('<div class="query-container">', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Enter your travel query and get flight and hotel information:</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        'Travel Query',
        height=200,
        key='query',
        placeholder='Type your travel query here...',
    )
    st.markdown('</div>', unsafe_allow_html=True)
    st.sidebar.image('/Users/mncedisimncwabe/Downloads/ai-travel.png', caption='AI Travel Assistant')

    return user_input


def process_query(user_input):
    if user_input:
        try:
            # Wrap the user query as HumanMessage in the Streamlit app
            messages = [HumanMessage(content=user_input)]
            # Pass message to the agent's graph
            result = st.session_state.agent.graph.invoke({'messages': messages})

            st.subheader('Travel Information')
            st.write(result['messages'][-1].content)
            
            # Store the result in session state in case needed later
            st.session_state.travel_info = result['messages'][-1].content

        except Exception as e:
            st.error(f'Error: {e}')
    else:
        st.error('Please enter a travel query.')


def main():
    initialize_agent()
    render_custom_css()
    user_input = render_ui()

    if st.button('Get Travel Information'):
        process_query(user_input)


if __name__ == '__main__':
    main()