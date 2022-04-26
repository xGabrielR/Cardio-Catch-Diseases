// Function To Load Google Scripts.
function OnOpen(){
  let ui = SpreadsheetApp.getUi();
  ui.createMenu('Cardiovascular Probability').addItem('Get Probability', 'PredictProba').addToUi();
};

let host_production = 'cardio-cath-d-app.herokuapp.com'

function ApiCall(data, endpoint){
  let url = "https://" + host_production + endpoint;
  let payload = JSON.stringify(data);
  let options = {'method': 'POST', 'contentType': 'application/json', 'payload': payload};

  let responde = UrlFetchApp.fetch(url, options);

  let status_code = responde.getResponseCode();
  let response_text = responde.getContentText();

  if(status_code !== 200){
    Logger.log('Response: (%s) %s', status_code, response_text)
  } else {
    pred = JSON.parse(response_text)
  };

  return pred

};

function PredictProba(){
  let active_sheet = SpreadsheetApp.getActiveSheet();
  let last_row = active_sheet.getLastRow();
  let title_columns = active_sheet.getRange('A1:L1').getValues()[0];

  let data = active_sheet.getRange('A2:'+'L'+last_row).getValues();

  for(row in data){
    let json = new Object();

    for(var j = 0; j < title_columns.length ; j++){
      json[title_columns[j]] = data[row][j];
    };

    let json_send = new Object;
    json_send['id']     = json['id']
    json_send['age']    = json['age']
    json_send['gender'] = json['gender']
    json_send['height'] = json['height']
    json_send['weight'] = json['weight']
    json_send['ap_hi']  = json['ap_hi']
    json_send['ap_lo']  = json['ap_lo']
    json_send['cholesterol'] = json['cholesterol']
    json_send['gluc']   = json['gluc']
    json_send['smoke']  = json['smoke']
    json_send['alco']   = json['alco']
    json_send['active'] = json['active']

    prediction = ApiCall(json_send, '/predict')
    
    active_sheet.getRange(Number(row)+2,13).setValue(prediction[0]['cardio_proba']);

  };

};
