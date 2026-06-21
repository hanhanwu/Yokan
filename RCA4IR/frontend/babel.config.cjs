// babel.config.cjs — CommonJS required because package.json has "type": "module"
module.exports = function (api) {
  api.cache(true);
  return {
    presets: ["babel-preset-expo"],
  };
};
