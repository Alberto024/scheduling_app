# scheduling_app
A streamlit app to help schedule shifts

# Installation

## Local Install
Install the required environment locally using (poetry)[https://python-poetry.org/]
```
poetry install
```

Then start a shell in the local environment
```
poetry shell
```

Then start the app with (streamlit)[https://www.streamlit.io/]
```
streamlit run app/scheduling_app.py
```

The app should now be visible at port `8501`, or another similar port if its not available 

## Docker install

To build the app
```
docker-compose build
```

To run the app
```
docker-compose up -d
```

To stop the app
```
docker-compose stop
```

After building and starting the app, it should now be visible at port 8000
